import torch
import torch.nn as nn
import torch.nn.functional as F
from model.fusion import get_fusion

class TIRG(nn.Module):
    def __init__(self, fusion, embed_dim=512):
        super(TIRG, self).__init__()
        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1, 1]))
        if fusion == 'base':
            concat_num = 2
        else:
            concat_num = 3

        self.gated_feature_composer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(concat_num * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(concat_num * embed_dim, embed_dim))
        self.res_info_composer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(concat_num * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(concat_num * embed_dim, 2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim))

        if fusion == 'hadamard':
            self.fusion = get_fusion(fusion)
        elif fusion == 'concat':
            self.fusion = get_fusion(fusion, embed_dim)
        elif fusion == 'base':
            self.fusion = None
        else: 
            #self.fusion = get_fusion(fusion, embed_dim, embed_dim, embed_dim, None, 12, 0.2)
            self.fusion = get_fusion(fusion, 256, embed_dim, embed_dim, None, 4, 0.2)

    def forward(self, imgs, texts):
        if len(texts.size()) > 2:
            texts = texts.squeeze(1)
        if self.fusion is None:
            x = torch.cat([imgs, texts], dim=1)
        else:
            fusion = self.fusion(imgs, texts)
            x = torch.cat([imgs, texts, fusion], dim=1)
        f1 = self.gated_feature_composer(x)
        f2 = self.res_info_composer(x)
        f = torch.sigmoid(f1) * imgs * self.a[0] + f2 * self.a[1]
        return f

class TIRG2(nn.Module):
    def __init__(self, fusion, embed_dim):
        super(TIRG2, self).__init__()
        self.tirg = TIRG(fusion, embed_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
    def forward(self, imgs, texts):
        imgs = self.fc1(imgs)
        texts = self.fc2(texts)
        return self.tirg(imgs, texts)

class CONCAT(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim)
        )
    
    def forward(self, src, trg):
        x = torch.cat((src, trg), -1) # B x 2D
        return self.fc(x)
    
class SUM(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
    
    def forward(self, src, trg):
        return src + trg

class VAL(nn.Module):
    def __init__(self, embed_dim):
        super(VAL, self).__init__()
        self.conv1 = nn.Conv2d(embed_dim * 2, embed_dim, 1)
        self.att_ch_conv = nn.Conv2d(embed_dim, embed_dim, 1)
        self.att_sp_conv = nn.Conv2d(1, 1, 3, padding=1)

        # the number of heads is tunable
        self.mh_att = nn.MultiheadAttention(embed_dim, num_heads=2, bias=False)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, 1)

        # weight parameter
        self.a = torch.nn.Parameter(torch.tensor([0.0, 1.0]))

    def forward(self, img_feat, text_feat):
        # img_feat : B x H x W x D
        # text_feat : B x D
        img_feat = img_feat.permute(0,3,1,2)
        B, D, H, W = img_feat.size()
        text_feat = text_feat.view(B, D, 1, 1).expand(-1, -1, H,
                                                      W)  # B x D x H x W
        v1_feat = torch.cat([img_feat, text_feat], 1)  # B x 2D x H x W
        v1_feat = self.conv1(v1_feat)  # B x D x H x W
        ##############################
        gate_sqz = v1_feat.mean((2, 3), keepdim=True)  # B x D x 1 x 1
        att_ch = self.att_ch_conv(gate_sqz)  # B x D x 1 x 1

        gate_sqz = v1_feat.mean(1, keepdim=True)  # B x 1 x H x W
        att_sp = self.att_sp_conv(gate_sqz)  # B x 1 x H x W

        joint_att = torch.sigmoid(att_ch) * torch.sigmoid(
            att_sp)  # B x D x H x W

        ##############################
        v1_feat = v1_feat.view(B, D, H * W).permute(2, 0, 1)  # H*W x B x D
        self_att, _ = self.mh_att(v1_feat, v1_feat, v1_feat)  # H*W x B x D
        self_att = self_att.view(H, W, B, D).permute(2, 3, 0,
                                                     1)  # B x D x H x W
        self_att = self.conv2(self_att)  # B x D x H x W

        ##############################
        composite_features = self.a[0] * joint_att * img_feat + self.a[
            1] * self_att
        composite_features = composite_features.mean((2, 3))  # B x D
        return composite_features
    
    
   
    
class NormalizationLayer(nn.Module):
    def __init__(self, normalize_scale=5.0, learn_scale=True):
        super(NormalizationLayer, self).__init__()
        self.norm_s = float(normalize_scale)
        if learn_scale:
            self.norm_s = torch.nn.Parameter(torch.FloatTensor((self.norm_s,)))

    def forward(self, x):
        features = self.norm_s * x / torch.norm(x, dim=-1, keepdim=True).expand_as(x)
        return features

class Sum(nn.Module):
    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, imgs, texts):
        if len(texts.size()) > 2:
            texts = texts.squeeze(1)
        f = imgs + texts
        return f


class DIFF(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim*3, embed_dim)
        )
    
    def forward(self, x_before, x_after):
        x_diff = x_after - x_before
        x = torch.cat((x_before, x_diff, x_after), -1) # B x 3*D
        return self.fc(x)

class FusDiffcg(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.cg1 = ContextGating(embed_dim, False)
        self.cg2 = ContextGating(embed_dim, False)
        self.fd = FusDiff(embed_dim)
        
    def forward(self, x_before, x_after):
        x_before = self.cg1(x_before)
        x_after = self.cg2(x_after)
        return self.fd(x_before, x_after)
    
class FusDiff(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(embed_dim*3, embed_dim),
            nn.ReLU()
        )
        
    def forward(self, x_before, x_after):
        #x_before = self.cg1(x_before)
        #x_after = self.cg2(x_after)
        
        x_before_ = self.fc1(torch.cat((x_before * x_after, x_before), -1)) # B x D
        x_after_ = self.fc2(torch.cat((x_before * x_after, x_after), -1)) # B x D
        x_diff = x_after_ - x_before_
        
        x = torch.cat((x_before, x_diff, x_after), -1) # B x 3*D
        x = self.fc(x)
        
        return x
    
class TIRGDiff(nn.Module):
    def __init__(self, fusion, embed_dim):
        super().__init__()
        self.tirg = TIRG(fusion, embed_dim)
        
        
    def forward(self, src, trg):
        return self.tirg(trg, src)
    
    
class DUDA(nn.Module):
    def __init__(self, embed_dim, modalities):
        super().__init__()
        self.modalities = modalities
        self.att_fc = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(embed_dim*3, embed_dim)
        
    def forward(self, src_experts, trg_experts):
        src, trg = [], []
        for m in self.modalities:
            src.append(src_experts[m])
            trg.append(trg_experts[m])
        src = torch.stack(src, 1) # B x M x D
        trg = torch.stack(trg, 1) # B x M x D
        diff = trg - src # B x M x D
        src_ = torch.cat([src, diff], -1) # B x M x 2D
        trg_ = torch.cat([trg, diff], -1) # B x M x 2D
        src_att = self.att_fc(src_) # B x M x 1
        trg_att = self.att_fc(trg_) # B x M x 1
        src__ = (src_att * src).sum(1) # B x D
        trg__ = (trg_att * trg).sum(1) # B x D
        x = torch.cat((src__, trg__-src__, trg__), -1)
        x = self.fc(x)
        
        return x
    
class DiffPool(nn.Module):
    def __init__(self, embed_dim, modalities):
        super().__init__()
        self.modalities = modalities
        self.fc = nn.Linear(embed_dim*3, embed_dim)
        
    def forward(self, src_experts, trg_experts):
        src, trg = [], []
        for m in self.modalities:
            src.append(src_experts[m])
            trg.append(trg_experts[m])
        src = torch.stack(src, 1).mean(1) # B x D
        trg = torch.stack(trg, 1).mean(1) # B x D
        diff = trg - src # B x D
        x = torch.cat((src, diff, trg), -1) # B x 3D
        x = self.fc(x)
        
        return x
    
    
class CONCATPool(nn.Module):
    def __init__(self, embed_dim, modalities):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.ReLU()
        )
        self.modalities = modalities
    
    def forward(self, src_experts, trg_experts):
        src, trg = [], []
        for m in self.modalities:
            src.append(src_experts[m])
            trg.append(trg_experts[m])
        src = torch.stack(src, 1).mean(1) # B x D
        trg = torch.stack(trg, 1).mean(1) # B x D
        x = torch.cat((src, trg), -1) # B x 2D
        return self.fc(x)
    
class FusPool(nn.Module):
    def __init__(self, embed_dim, modalities):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
        self.modalities = modalities
    
    def forward(self, src_experts, trg_experts):
        src, trg = [], []
        for m in self.modalities:
            src.append(src_experts[m])
            trg.append(trg_experts[m])
        src = torch.stack(src, 1).mean(1) # B x D
        trg = torch.stack(trg, 1).mean(1) # B x D
        x = src * trg # B x 2D
        return self.fc(x)


class TIRGDiffPool(nn.Module):
    def __init__(self, fusion, embed_dim, modalities):
        super().__init__()
        self.tirg = TIRG(fusion, embed_dim)
        self.modalities = modalities
    def forward(self, src_experts, trg_experts):
        src, trg = [], []
        for m in self.modalities:
            src.append(src_experts[m])
            trg.append(trg_experts[m])
        src = torch.stack(src, 1).mean(1) # B x D
        trg = torch.stack(trg, 1).mean(1) # B x D
        return self.tirg(trg, src)
    
      
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
