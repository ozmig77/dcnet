import sys
from collections import OrderedDict, defaultdict
import time
import torch
import torch.nn.functional as F
import numpy as np
from base import BaseTrainer
from model.metric import sharded_cross_view_inner_product
from model.metric import compute_score


class TrainerJoint(BaseTrainer):

    def __init__(self, model, loss, optimizer, config, data_loaders, lr_scheduler):
        super().__init__(model, loss, optimizer, config)
        self.config = config
        self.data_loaders = data_loaders
        self.lr_scheduler = lr_scheduler
        if "train" in self.data_loaders.dataloaders:
            self.len_epoch = len(self.data_loaders["train"])
            self.log_step = int(np.sqrt(len(data_loaders["train"].dataset)))

    def _train_epoch(self, epoch, mode="train"):
        self.model.train()
        total_loss = 0
        progbar = Progbar(len(self.data_loaders[mode].dataset))
        modalities = self.model.image_encoder.modalities
        for batch_idx, batch in enumerate(self.data_loaders[mode]):
            for experts in ['candidate_experts', 'target_experts']:
                for key, val in batch[experts].items():
                    batch[experts][key] = val.to(self.device)
            batch["text"] = batch["text"].to(self.device)
            B, M = len(batch['text']), len(self.model.expert_dims)
            self.optimizer.zero_grad()
            # {mod: B x D}
            ref_experts = self.model.get_image_emb(batch['candidate_experts'], istgt=False)
            trg_experts = self.model.get_image_emb(batch['target_experts'], istgt=True)
            # {mod: B x D}
            text_experts = self.model.get_text_emb(batch['text'],
                                                   batch['text_lengths'])
            
            # --- composition --------------------------------------
            mod_trg = self.model.get_composition(ref_experts, text_experts)
            
            score_matrix_comp = self.model.get_score_matrix(
                trg_embds=trg_experts, # {mod: N x D}
                src_embds=mod_trg, # {mod: B x D} 
                subspaces=self.model.modalities,
            )
           
            # --- correction ---------------------------------
            # {mod: B x N x D}
            if self.model.feat2d:
                ref_experts_new = {}
                for m in ref_experts:
                    ref_experts_new[m] = ref_experts[m].mean((1,2))
            else:
                ref_experts_new = ref_experts
            mod_text = self.model.get_correction(ref_experts_new, trg_experts)
            
            score_matrix_corr = self.model.get_score_matrix(
                trg_embds=mod_text, # {mod: B x N x D or N x D}
                src_embds=text_experts, # {mod: B x D} 
                subspaces=self.model.modalities,
            )
            
            # loss
            loss = self.loss(score_matrix_comp) + self.loss(score_matrix_corr)
            
            # d-trg loss
            
            if self.config['arch']['joint_loss'] != "":
                if epoch > self.config['arch']['joint_epoch'] :
                    d_mod_text = {}
                    for mod in mod_text:
                        d_mod_text[mod] = torch.diagonal(mod_text[mod], 
                                                         dim1=0, 
                                                         dim2=1).transpose(0,1) # N x D
                    if self.config['arch']['joint_loss'] in ["cyc3","cyc3cyc"]:
                        d_mod_text_ = {}
                        for mod in mod_text:
                            d_mod_text_[mod] = (d_mod_text[mod] + text_experts[mod])/2
                        aux_trg = mod_trg
                        aux_src = self.model.get_composition(ref_experts, d_mod_text_)
                    elif self.config['arch']['joint_loss'] == "cyc3_1":
                        aux_trg = mod_trg
                        aux_src = self.model.get_composition(ref_experts, d_mod_text)
                    elif self.config['arch']['joint_loss'] == "cyc3d":
                        for mod in mod_text:
                            d_mod_text[mod] = (d_mod_text[mod] + text_experts[mod])/2
                            d_mod_text[mod] = d_mod_text[mod].detach()
                        aux_trg = mod_trg
                        aux_src = self.model.get_composition(ref_experts, d_mod_text)
                    score_matrix_joint = self.model.get_score_matrix(
                        trg_embds=aux_trg, # {mod: B x N x D or N x D}
                        src_embds=aux_src, # {mod: B x D} 
                        subspaces=self.model.modalities,
                    )
                    joint_loss = self.loss(score_matrix_joint)
                    
                    if self.config['arch']['joint_loss'] == "cyc3cyc":
                        aux2_trg = self.model.get_correction(ref_experts_new, mod_trg)
                        aux2_src = d_mod_text_
                        score_matrix_joint2 = self.model.get_score_matrix(
                            trg_embds=aux2_trg, # {mod: B x N x D or N x D}
                            src_embds=aux2_src, # {mod: B x D} 
                            subspaces=self.model.modalities,
                        )
                        joint_loss += self.loss(score_matrix_joint2)
                    
                    loss += self.config['arch']['joint_weight'] * joint_loss
            
            loss.backward()
            self.optimizer.step()

            progbar.add(B, values=[('loss', loss.item())])
            total_loss += loss.item()

        if mode == "train":
            log = {'loss': total_loss / self.len_epoch}

            if epoch > self.val_epoch: 
                val_log = self._valid_epoch()
                log.update(val_log)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        else:
            log = None

        return log

    def _valid_epoch(self, save_textatt = False):
    
        
        self.model.eval()
        categories = self.config['data_loader']['args']['categories']
        metric = {'recall': np.zeros((len(categories), 4)), 'score': dict(),
                  'largest': self.largest, 'comb':np.zeros((len(categories),2))}
        modalities = self.data_loaders[categories[0]].dataset.ordered_experts
        if save_textatt:
            print("save att")
            text_att = {cat : [] for cat in categories}
        for i, category in enumerate(categories):
            # ---------- pre-build target features ------------
            val_experts = {expert: list() for expert in modalities}
            id2targetidx = {}
            for batch in self.data_loaders[category + '_trg']:
                # Features
                for elem in batch['meta_info']:
                    id2targetidx[elem['candidate']] = len(id2targetidx)
                for key, val in batch['candidate_experts'].items():
                    batch['candidate_experts'][key] = val.to(self.device)

                with torch.no_grad():
                    experts = self.model.get_image_emb(batch['candidate_experts'], istgt=True)
                    for modality, val in experts.items():
                        val_experts[modality].append(val)
            for modality, val in val_experts.items():
                val_experts[modality] = torch.cat(val) # N x D
                
            # -------- Start evaluation -------------------------
            score_comp, score_corr = [], []
            meta_infos = []
            N = val_experts[modality].size(0)
            for batch in self.data_loaders[category]:
                
                for experts in ['candidate_experts']:
                    for key, val in batch[experts].items():
                        batch[experts][key] = val.to(self.device)
                batch["text"] = batch["text"].to(self.device)
                B, M = len(batch['text']), len(self.model.expert_dims)
                meta_infos.extend(list(batch['meta_info']))
                with torch.no_grad():
                    # {mod: B x D}
                    src_experts = self.model.get_image_emb(batch['candidate_experts'], istgt=False)
                    # {mod: B x D}
                    text_experts = self.model.get_text_emb(batch['text'],
                                                           batch['text_lengths'])
                    if save_textatt:
                        tmp_att = self.model.text_encoder.shared_att # B x M x L 
                        text_att[category].append((list(batch['meta_info']), 
                            tmp_att.cpu()
                        ))

                    # --- Composition --------------------------------------
                    mod_trg = self.model.get_composition(src_experts, text_experts)
                    score_matrix_comp = self.model.get_score_matrix(
                        trg_embds=val_experts, # {mod: N x D}
                        src_embds=mod_trg, # {mod: B x D} 
                        subspaces=self.model.modalities,
                        l2renorm=False,#True,
                        dist=False,#True,
                        val=True
                    )

                    # --- Correction ---------------------------------
                    # {mod: B x N x D}
                    if self.model.feat2d:
                        src_experts_new = {}
                        for m in src_experts:
                            src_experts_new[m] = src_experts[m].mean((1,2))
                    else:
                        src_experts_new = src_experts
                    mod_text = self.model.get_correction(src_experts_new, val_experts)

                    score_matrix_corr = self.model.get_score_matrix(
                        trg_embds=mod_text, # {attr0: B x N x D}
                        src_embds=text_experts, # {attr0: B x D}
                        subspaces=self.model.modalities,
                        l2renorm=False,#True,
                        dist=False,#True,
                        val=True
                    )
                    # Delete reference == target
                    for ei, elem in enumerate(batch['meta_info']):
                        score_matrix_comp[ei, id2targetidx[elem['candidate']]] = -10e10 
                        score_matrix_corr[ei, id2targetidx[elem['candidate']]] = -10e10 
                    # --- Sum --------------
                    score_comp.append(score_matrix_comp)
                    score_corr.append(score_matrix_corr)
           
            score_comp = torch.cat(score_comp)
            score_corr = torch.cat(score_corr)
            val_ids = self.data_loaders[category + '_trg'].dataset.data
            
            for j, score in enumerate(score_comp):
                _, topk = score.topk(dim=0, k=50, largest=self.largest, sorted=True)
                meta_infos[j]['ranking'] = [val_ids[idx] for idx in topk]
            r10_1, r50_1 = compute_score(meta_infos, meta_infos)
            for j, score in enumerate(score_corr):
                _, topk = score.topk(dim=0, k=50, largest=self.largest, sorted=True)
                meta_infos[j]['ranking'] = [val_ids[idx] for idx in topk]
            r10_2, r50_2 = compute_score(meta_infos, meta_infos)
            metric['recall'][i] = r10_1, r50_1, r10_2, r50_2

            # Combine
            for j in range(len(score_comp)):
                score_comb = F.log_softmax(score_comp[j], dim=-1) + F.log_softmax(score_corr[j], dim=-1) # N
                _, topk = score_comb.topk(dim=0, k=50, largest=self.largest, sorted=True)
                meta_infos[j]['ranking'] = [val_ids[idx] for idx in topk]
            r10_c, r50_c = compute_score(meta_infos, meta_infos)
            metric['comb'][i] = r10_c, r50_c
                
            metric['score'][category] = {'ids': val_ids, 'matrix_comp': score_comp.cpu(), 'matrix_corr':score_corr.cpu(), 'meta_info': meta_infos}

        metric['recall_avg'] = metric['recall'][:,:2].mean()
        metric['recall_avg_corr'] = metric['recall'][:,2:].mean()
        metric['comb_avg'] = metric['comb'].mean()
        if save_textatt:
            save_path = self.config.save_dir / f'att_result.pt'
            torch.save(text_att, save_path)
        return metric


class Progbar:
    def __init__(self, target, width=30, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        self._values = OrderedDict()
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        values = values or []
        for k, v in values:
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)