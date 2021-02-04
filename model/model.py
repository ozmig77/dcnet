import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import model.composition as cdmodels
from model.composition import NormalizationLayer
from model.modules import get_text_pooling, masked_softmax
from model.metric import InnerProduct, ExtendedProduct
from base import BaseModel
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CENet(BaseModel):
    def __init__(self, text_dim, composition, correction, fusion,
                 expert_dims, same_dim, text_feat, norm_scale=1.0,
                 text_moe=False, backbone_pretrain='none',
                 backbone='densenet', text_pooling_module='sharedconv',
                 metric_fn='ip', ref=False):
        super(CENet, self).__init__()

        self.modalities = list(expert_dims.keys())
        self.composition_type = composition
        self.correction_type = correction
        self.fusion = fusion
        self.expert_dims = expert_dims
        self.text_feat = text_feat
        self.feat2d = self.composition_type in ['val']
        
        # Encoder        
        self.text_encoder = TextCEModuleNew(
            expert_dims=expert_dims,
            text_dim=text_dim,
            same_dim=same_dim,
            text_pooling_module=text_pooling_module
        )

        self.image_encoder = ImageCEModule(
            backbone=backbone,
            expert_dims=expert_dims,
            same_dim=same_dim,
            backbone_pretrain=backbone_pretrain,
            feat2d=self.feat2d
        )
        
        # Normalization
        norm_list = [
            [mode, NormalizationLayer(normalize_scale=norm_scale, learn_scale=True)]
            for mode in ['image', 'text']
        ]
        self.norm_layer = nn.ModuleDict(norm_list)
        
        # Composition & Correction
        if self.composition_type == 'tirg':
            composition_list = [cdmodels.TIRG(fusion, embed_dim=same_dim) for _ in self.expert_dims]
            self.composition_layer = nn.ModuleList(composition_list)
        if self.composition_type == 'tirg2':
            composition_list = [cdmodels.TIRG2(fusion, embed_dim=same_dim) for _ in self.expert_dims]
            self.composition_layer = nn.ModuleList(composition_list)
        if self.composition_type == 'val':
            composition_list = [cdmodels.VAL(embed_dim=same_dim) for _ in self.expert_dims]
            self.composition_layer = nn.ModuleList(composition_list)
        if self.composition_type == 'concat':
            composition_list = [cdmodels.CONCAT(embed_dim=same_dim) for _ in self.expert_dims]
            self.composition_layer = nn.ModuleList(composition_list)
        if self.composition_type == 'sum':
            composition_list = [cdmodels.SUM(embed_dim=same_dim) for _ in self.expert_dims]
            self.composition_layer = nn.ModuleList(composition_list)
        if self.correction_type == 'base':
            correction_list = [cdmodels.Diff(embed_dim=same_dim) for _ in self.expert_dims]
            self.correction_layer = nn.ModuleList(correction_list)
        if self.correction_type == 'fd':
            correction_list = [cdmodels.FusDiff(embed_dim=same_dim) for _ in self.expert_dims]
            self.correction_layer = nn.ModuleList(correction_list)
        if self.correction_type == 'fdcg':
            correction_list = [cdmodels.FusDiffcg(embed_dim=same_dim) for _ in self.expert_dims]
            self.correction_layer = nn.ModuleList(correction_list)
        if self.correction_type == 'diff':
            correction_list = [cdmodels.DIFF(embed_dim=same_dim) for _ in self.expert_dims]
            self.correction_layer = nn.ModuleList(correction_list)
        if self.correction_type == 'tirg':
            correction_list = [cdmodels.TIRGDiff(fusion, same_dim) for _ in self.expert_dims]
        if self.correction_type == 'tirgori':
            correction_list = [cdmodels.TIRG(fusion, same_dim) for _ in self.expert_dims]
            self.correction_layer = nn.ModuleList(correction_list)
        if self.correction_type == 'tirgpool':
            self.correction_layer = cdmodels.TIRGDiffPool(fusion, same_dim, self.modalities)
        if self.correction_type == 'concat':
            self.correction_layer = cdmodels.CONCATPool(same_dim, self.modalities)
        if self.correction_type == 'duda':
            self.correction_layer = cdmodels.DUDA(same_dim, self.modalities)
        if self.correction_type == 'diffpool':
            self.correction_layer = cdmodels.DiffPool(same_dim, self.modalities)
        if self.correction_type == 'fuspool':
            self.correction_layer = cdmodels.FusPool(same_dim, self.modalities)
        if ref:
            ref_list = [cdmodels.FusDiff(embed_dim=same_dim) for _ in self.expert_dims]
            self.ref_layer = nn.ModuleList(ref_list)
        
        # ----- Scoring module -------------------------
        if metric_fn == 'ip':
            self.get_score_matrix = InnerProduct()        
        elif metric_fn == 'ext':
            self.get_score_matrix = ExtendedProduct() 
    def get_norm(self, experts, mode):
        # experts : {B x D}
        # mode: str, "image" or "text"
        for mod in self.modalities:
            experts[mod] = self.norm_layer[mode](experts[mod]) # -1 x D
        return experts
    
    def get_image_emb(self, inputs, istgt):
        experts = self.image_encoder(inputs)
        if self.feat2d:
            if istgt:
                for mod in self.modalities:
                    experts[mod] = experts[mod].mean((1,2)) # B x D
            else:
                return experts
        return self.get_norm(experts, 'image')
    
    def get_text_emb(self, text, text_len):
        experts = self.text_encoder(text, text_len)
        return self.get_norm(experts, 'text')

        
    def get_correction(self, src_experts, trg_experts):
        # src : {B x D}
        # trg : {N x D}
        # return : {B x N x D}
        B = src_experts[self.modalities[0]].size(0)
        N = trg_experts[self.modalities[0]].size(0)
        diff_feature = defaultdict(list)
        for bi in range(B):
            # {mod: N x D}
            new_src_expert = {mod: src_experts[mod][bi].unsqueeze(0).expand(N,-1) 
                                for mod in self.modalities}
            if self.correction_type in ['duda', 'diffpool','tirgpool','concat','fdpool','fuspool']:
                diff_feature[self.modalities[0]].append(
                    self.norm_layer['text'](
                        self.correction_layer(new_src_expert, trg_experts)
                   )
                )
            else:
                # {mod: N x D}
                for mod, layer in zip(self.modalities, self.correction_layer):
                    diff_feature[mod].append(
                        self.norm_layer['text'](layer(new_src_expert[mod], trg_experts[mod]))
                    )
        for mod in self.modalities:
            diff_feature[mod] = torch.stack(diff_feature[mod], 0) # B x N x D
            if self.correction_type in ['duda', 'diffpool','tirgpool','concat','fdpool','fuspool']:
                break
        return diff_feature
    
    
    def get_composition(self, src, text):
        # src : {B x D}
        # text : {B x D}
        # return : {B x D}
        result = {}
        for mod, layer in zip(self.modalities, self.composition_layer):
            res = layer(src[mod], text[mod]) # B x D
            res = self.norm_layer['image'](res)
            result[mod] = res # B x D
        return result # B x D

    def get_ref(self, src_experts, trg_experts):
        # src : {B x D}
        # trg : {N x D}
        # return : {B x N x D}
        B = src_experts[self.modalities[0]].size(0)
        N = trg_experts[self.modalities[0]].size(0)
        diff_feature = defaultdict(list)
        for bi in range(B):
            # {mod: N x D}
            new_src_expert = {mod: src_experts[mod][bi].unsqueeze(0).expand(N,-1) 
                                for mod in self.modalities}
            # {mod: N x D}
            for mod, layer in zip(self.modalities, self.ref_layer):
                diff_feature[mod].append(
                    self.norm_layer['image'](layer(new_src_expert[mod], trg_experts[mod]))
                    )
        for mod in self.modalities:
            diff_feature[mod] = torch.stack(diff_feature[mod], 0) # B x N x D
        return diff_feature
    
class TextCEModuleNew(nn.Module):
    def __init__(self, expert_dims, text_dim, same_dim, text_pooling_module=''):
        super(TextCEModuleNew, self).__init__()

        modalities = list(expert_dims.keys())
        self.expert_dims = expert_dims
        self.modalities = modalities
        num_mods = len(expert_dims)
        self.moe_weights = torch.ones(1, num_mods) / num_mods
        self.text_pooling_module = text_pooling_module
        if self.text_pooling_module in ['sharedconv', 'sharedfc', 'sharedgru']:
            self.text_encoder = get_text_pooling(text_pooling_module,
                                        word_dim=text_dim, # Text
                                        hidden_dim=same_dim)
            encoder_text_dim = self.text_encoder.out_dim
            self.modality_embedding = nn.Parameter(
                torch.zeros((1, len(modalities), encoder_text_dim))
            )
            nn.init.xavier_uniform_(self.modality_embedding)
            self.text_att = nn.Sequential(
                nn.Linear(encoder_text_dim, encoder_text_dim),
                nn.ReLU(),
                nn.Linear(encoder_text_dim, 1)
            )
        elif self.text_pooling_module in ['singleconv']:
            self.text_encoder = get_text_pooling(text_pooling_module,
                                        word_dim=text_dim,
                                        hidden_dim=same_dim)
            encoder_text_dim = self.text_encoder.out_dim
        else:
            # Pooling
            text_pooling_list = [
                    [mode, get_text_pooling(text_pooling_module,
                                            word_dim=text_dim,
                                            hidden_dim=same_dim)]
                       for mode in modalities]
            self.text_pooling = nn.ModuleDict(text_pooling_list)
            encoder_text_dim = self.text_pooling[modalities[0]].out_dim

        # Experts
        text_embed_dict = {}
        for key in modalities:
            text_embed_dict[key] = nn.Sequential(
              nn.Linear(encoder_text_dim, same_dim),
              nn.ReLU() # NoReLU
            )
            #text_embed_dict[key] = GatedEmbeddingUnit(encoder_text_dim, same_dim, use_bn=True)
        self.text_embed = nn.ModuleDict(text_embed_dict)

    def forward(self, text, text_lengths):
        batch_size, max_words, text_feat_dim = text.size()
        B, M = text.size(0), len(self.modalities)
        if self.text_pooling_module in ['sharedconv', 'sharedfc', 'sharedgru']:
            agg_feat, mask = self.text_encoder(text, text_lengths=text_lengths) # B x L x D, B x L
            att_x = self.modality_embedding.unsqueeze(2) * agg_feat.unsqueeze(1) # B x M x L x D
            att_x = self.text_att(att_x) # B x M x L x 1
            att_x = masked_softmax(att_x.squeeze(-1), mask.unsqueeze(1)) # B x M x L
            self.shared_att = att_x
            att_feat = att_x.unsqueeze(-1) * agg_feat.unsqueeze(1) # B x M x L x D
            att_feat = att_feat.sum(2) # B x M x D
        elif self.text_pooling_module in ['singleconv']:
            agg_feat, mask = self.text_encoder(text, text_lengths=text_lengths) # B x L x D, B x L
            agg_feat = agg_feat.sum(1) / mask.sum(1, keepdim=True).clamp(min=1) # B x D
             
        text_embd = {}
        for midx, modality in enumerate(self.modalities):
            if self.text_pooling_module in ['sharedconv','sharedfc', 'sharedgru']:
                text_emb = att_feat[:,midx,:] # B x D
            elif self.text_pooling_module in ['singleconv']:
                text_emb = agg_feat
            else:
                text_emb = self.text_pooling[modality](text, 
                                                       text_lengths=text_lengths) # B x D

            text_ = self.text_embed[modality](text_emb) # B x D
            text_embd[modality] = text_ # B x D
                   
        return text_embd
    

class ImageCEModule(nn.Module):
    def __init__(self, backbone, expert_dims, backbone_pretrain, same_dim=512, feat2d=False):
        super(ImageCEModule, self).__init__()

        modalities = list(expert_dims.keys())
        self.expert_dims = expert_dims
        self.modalities = modalities
        self.backbone_name = backbone
        self.feat2d = feat2d

        # -------- Backbone --------------------------------------------------- 
        if backbone == 'resnet':
            resnet = models.resnet50(pretrained=True)
            modules = list(resnet.children())[:-2]
            self.backbone = nn.Sequential(*modules)
        elif backbone == 'resnet152':
            resnet = models.resnet152(pretrained=True)
            modules = list(resnet.children())[:-2]
            self.backbone = nn.Sequential(*modules)
        elif backbone == 'densenet':
            densenet = models.densenet169(pretrained=True)
            modules = list(densenet.children())[:-1]
            self.backbone = nn.Sequential(*modules)
        else:
            raise ValueError
        ## Load deepfashion pretrained backbone
        if backbone_pretrain == "deepfashion":
            print("Load %s_%s as backbone"%(backbone_pretrain, backbone))
            param_dict = torch.load('deepfashion/logdir/%s_%s/best_ckpt.pth.tar'%(backbone_pretrain, backbone))['state_dict']
            for i in param_dict:
                if "backbone." in i:
                    self.backbone.state_dict()[i[9:]].copy_(param_dict[i])
        if backbone == 'densenet':
            self.backbone = self.backbone[0]
        self.dropout = nn.Dropout(p=0.2)
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        # ------ Embedding for modalities -------------------------------------
        vis_embed_dict = {}
        for key in modalities:
            vis_embed_dict[key] = nn.Sequential(
              nn.Linear(expert_dims[key], same_dim),
              nn.ReLU() # NoReLU
            )
            #vis_embed_dict[key] = ReduceDim(expert_dims[key], same_dim)
        self.vis_embed = nn.ModuleDict(vis_embed_dict)

        #gated_vid_embds = {mod: MimicCEGatedEmbeddingUnit(same_dim, same_dim, use_bn=True) for mod in modalities}
        #self.video_GU = nn.ModuleDict(gated_vid_embds)

            
    def forward(self, inputs):
        B = inputs[self.modalities[0]].size(0)
        # ---- Get feature from Backbone ---------------------------------------
        experts = {}
        x = inputs['im_feat']
        for ii, module in enumerate(self.backbone):
            x = module(x)
            if self.backbone_name == 'densenet': # Get intermediate feature
                if ii == 7 and "inter-2" in self.modalities:
                    experts['inter-2'] = x
                if ii == 9 and "inter-1" in self.modalities:
                    experts['inter-1'] = x
            if self.backbone_name in ['resnet', 'resnet152']:
                if ii == 5 and "inter-2" in self.modalities:
                    experts['inter-2'] = x
                if ii == 6 and "inter-1" in self.modalities:
                    experts['inter-1'] = x
        image_feature = x
        pooled_im_feat = self.dropout(self.avg_pooling(image_feature).squeeze(-1).squeeze(-1))
        
        # ---- Embedding -------------------------------------
        for mod in self.modalities:
            if mod == 'im_feat':
                if self.feat2d:
                    val = self.dropout(image_feature).permute(0,2,3,1) # B x H x W x D
                else:
                    val = pooled_im_feat                    
            elif "inter" in mod:
                if self.feat2d:
                    val = self.dropout(experts[mod]).permute(0,2,3,1) # B x H x W x D                    
                else:
                    val = self.dropout(self.avg_pooling(experts[mod]).squeeze(-1).squeeze(-1))
            elif "spatial" in mod:
                if mod == 'spatial0':
                    hs,he,ws,we = 1,2,3,4 #0,3,2,5
                if mod == 'spatial1':
                    hs,he,ws,we, = 3,4,1,2 #2,5,0,3
                if mod == 'spatial2':
                    hs,he,ws,we, = 3,4,3,4 #2,5,2,5
                if mod == 'spatial3':
                    hs,he,ws,we, = 3,4,5,6 #2,5,4,7
                if mod == 'spatial4':
                    hs,he,ws,we, = 5,6,3,4 #4,7,2,5
                val = self.dropout(self.avg_pooling(image_feature[:,:,hs:he,ws:we]).squeeze(-1).squeeze(-1))
            experts[mod] = self.vis_embed[mod](val)
            
        #for mod in self.modalities:
        #    experts[mod] = self.video_GU[mod](experts[mod])
        return experts
            

class GatedEmbeddingUnit(nn.Module):
    def __init__(self, input_dimension, output_dimension, use_bn):
        super(GatedEmbeddingUnit, self).__init__()

        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = ContextGating(output_dimension, add_batch_norm=use_bn)

    def forward(self, x):
        x = self.fc(x)
        x = self.cg(x)
        x = F.normalize(x)
        return x


class MimicCEGatedEmbeddingUnit(nn.Module):
    def __init__(self, input_dimension, output_dimension, use_bn):
        super(MimicCEGatedEmbeddingUnit, self).__init__()
        self.cg = ContextGating(input_dimension, add_batch_norm=use_bn)

    def forward(self, x):
        x = self.cg(x)
        x = F.normalize(x)
        return x


class ReduceDim(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(ReduceDim, self).__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)

    def forward(self, x):
        x = self.fc(x)
        x = F.normalize(x)
        return x

class ContextGating(nn.Module):
    def __init__(self, dimension, add_batch_norm=True):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(dimension)

    def forward(self, x):
        x1 = self.fc(x)
        if self.add_batch_norm:
            x1 = self.batch_norm(x1)
        x = torch.cat((x, x1), 1)
        return F.glu(x, 1)
