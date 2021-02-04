import sys
from collections import OrderedDict, defaultdict
import time
import torch
import torch.nn.functional as F
import numpy as np
from base import BaseTrainer
from model.metric import sharded_cross_view_inner_product
from model.metric import compute_score


class TrainerRef(BaseTrainer):

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
            # {mod: B x D}, [B x A] (len: M)
            ref_experts = self.model.get_image_emb(batch['candidate_experts'], istgt=False)
            trg_experts = self.model.get_image_emb(batch['target_experts'], istgt=True)
           
            # {mod: B x D}, B x M
            text_experts = self.model.get_text_emb(batch['text'],
                                                   batch['text_lengths'])
            # --- ref --------------------------------------
            mod_ref = self.model.get_ref(text_experts, trg_experts)
            score_matrix_ref = self.model.get_score_matrix(
                trg_embds=mod_ref, # {mod: B x N x D or N x D}
                src_embds=ref_experts, # {mod: B x D} 
                subspaces=list(mod_ref.keys()),
            )
           
            loss = [self.loss(score_matrix_ref)]
            
            if self.config['trainer']['train_type'] in ['refcomp','refjoint']:
                # --- composition --------------------------------------
                mod_trg = self.model.get_composition(ref_experts, text_experts)
                score_matrix_comp = self.model.get_score_matrix(
                    trg_embds=trg_experts, # {mod: N x D}
                    src_embds=mod_trg, # {mod: B x D} 
                    subspaces=self.model.modalities,
                )
                loss.append(self.loss(score_matrix_comp))
                
                if self.config['trainer']['train_type'] == 'refjoint':
                    mod_text = self.model.get_correction(ref_experts, trg_experts)
                    score_matrix_corr = self.model.get_score_matrix(
                        trg_embds=mod_text, # {mod: B x N x D or N x D}
                        src_embds=text_experts, # {mod: B x D} 
                        subspaces=self.model.modalities,
                    )
                    loss.append(self.loss(score_matrix_corr))
            
            loss = sum(loss) #/ len(loss)
                    
            loss.backward()
            self.optimizer.step()

            progbar.add(B, values=[('loss', loss.item())])
            total_loss += loss.item()

        if mode == "train":
            log = {'loss': total_loss / self.len_epoch}

            if epoch > self.val_epoch: ###1
                val_log = self._valid_epoch()
                log.update(val_log)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        else:
            log = None

        return log

    def _valid_epoch(self):
        
        self.model.eval()
        categories = self.config['data_loader']['args']['categories']
        metric = {'recall': np.zeros((3, 2)), 'score': dict(), 'largest': self.largest}
        modalities = self.data_loaders[categories[0]].dataset.ordered_experts

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
                val_experts[modality] = torch.cat(val)
            # -------- Start evaluation -------------------------
            scores = []
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
                    # {mod: B x D}, [B x A] (len: M)
                    ref_experts = self.model.get_image_emb(batch['candidate_experts'], istgt=False)
                    # {mod: B x D}, B x M
                    text_experts = self.model.get_text_emb(batch['text'],
                                                           batch['text_lengths'])
                    
                    score_tot = 0
                    # --- Ref --------------------------------------
                    mod_ref = self.model.get_ref(text_experts, val_experts)
                    score_tot += self.model.get_score_matrix(
                        trg_embds=mod_ref, # {mod: B x N x D}
                        src_embds=ref_experts, # {mod: B x D}
                        subspaces=list(mod_ref.keys()),
                        l2renorm=False,#True,
                        dist=False,#True,
                        val=True
                    )
                    if self.config['trainer']['train_type'] in ['refcomp','refjoint']:
                        # --- Composition --------------------------------------
                        mod_trg = self.model.get_composition(ref_experts, text_experts)
                        score_tot += self.model.get_score_matrix(
                            trg_embds=val_experts, # {mod: N x D}
                            src_embds=mod_trg, # {mod: B x D} 
                            subspaces=self.model.modalities,
                            l2renorm=False,#True,
                            dist=False,#True,
                            val=True
                        )
                        if self.config['trainer']['train_type'] in ['refjoint']:
                            # --- Correction ---------------------------------
                            # {mod: B x N x D}
                            mod_text = self.model.get_correction(ref_experts, val_experts)
                            score_tot += self.model.get_score_matrix(
                                trg_embds=mod_text, # {attr0: B x N x D}
                                src_embds=text_experts, # {attr0: B x D}
                                subspaces=self.model.modalities,
                                l2renorm=False,#True,
                                dist=False,#True,
                                val=True
                            )
                    scores.append(score_tot)
           
            scores = torch.cat(scores)
            val_ids = self.data_loaders[category + '_trg'].dataset.data

            for j, score in enumerate(scores):
                _, topk = score.topk(dim=0, k=50, largest=self.largest, sorted=True)
                meta_infos[j]['ranking'] = [val_ids[idx] for idx in topk]

            r1, r10, r50 = compute_score(meta_infos, meta_infos)
            metric['recall'][i] = r10, r50
            if category == 'shoe':
                metric['recall'][1] = r1, 0
                
            metric['score'][category] = {'ids': val_ids, 'matrix': scores.cpu(), 'meta_info': meta_infos}

        metric['recall_avg'] = metric['recall'].mean()
        metric['comb_avg'] = metric['recall_avg']
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
