import torch
from abc import abstractmethod
import re
import time
from numpy import inf


class BaseTrainer:

    def __init__(self, model, loss, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer')

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.loss = loss
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.include_optim_in_ckpts = cfg_trainer.get('include_optim_in_ckpts', True)
        self.skip_first_n_saves = cfg_trainer.get('skip_first_n_saves', 0)
        self.num_keep_ckpts = cfg_trainer.get('num_keep_ckpts', 5)
        self.largest = cfg_trainer.get('largest', True)
        self.val_epoch = cfg_trainer.get('val_epoch', 5)
        self.categories = config['data_loader']['args']['categories']

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = config.save_dir

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch, mode):
        raise NotImplementedError

    def train(self):

        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch, mode="train")

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            if epoch > self.val_epoch:
                # print logged informations to the screen
                for key, value in log.items():
                    if key == 'recall_avg':
                        self.logger.info(f'[Avg Recall]     : {value}')
                    elif key == 'recall_avg_corr':
                        self.logger.info(f'[Avg Recall corr]: {value}')
                    elif key == 'comb_avg':
                        self.logger.info(f'[comb_avg]       : {value}')
                    elif key == 'recall':
                        for i, category in zip(value, self.categories):
                            if len(i) == 2:
                                self.logger.info(f'[{category}] r@10, r@50: {i[0]}\t{i[1]}')
                            elif len(i) == 4:
                                self.logger.info(f'[{category}] comp corr r@10, r@50: {i[0]}\t{i[1]}\t{i[2]}\t{i[3]}')
                    elif key == 'comb':
                        combstr = "comb:"
                        for i, category in zip(value, self.categories):
                            combstr += f' {i[0]} {i[1]}'
                        self.logger.info(combstr)
               
            # eval model according to configured metric, save best # ckpt as trained_model
            best = False
            not_improved_count = 0

            if epoch > self.val_epoch:
                if self.mnt_mode != 'off':
                    try:
                        # check whether specified metric improved or not, according to
                        # specified metric(mnt_metric)
                        lower = log[self.mnt_metric] <= self.mnt_best
                        higher = log[self.mnt_metric] >= self.mnt_best
                        improved = (self.mnt_mode == 'min' and lower) or \
                                   (self.mnt_mode == 'max' and higher)
                    except KeyError:
                        msg = "Warning: Metric '{}' not found, perf monitoring is disabled."
                        self.logger.warning(msg.format(self.mnt_metric))
                        self.mnt_mode = 'off'
                        improved = False
                        not_improved_count = 0

                    if improved:
                        self.mnt_best = log[self.mnt_metric]
                        not_improved_count = 0
                        best = True
                    else:
                        not_improved_count += 1

                    if not_improved_count > self.early_stop:
                        self.logger.info("Val performance didn\'t improve for {} epochs. "
                                         "Training stops.".format(self.early_stop))
                        break
            # If checkpointing is done intermittently, still save models that outperform
            # the best metric
            save_best = best and not self.mnt_metric == "epoch"

            # Due to the fast runtime/slow HDD combination, checkpointing can dominate
            # the total training time, so we optionally skip checkpoints for some of
            # the first epochs
            if epoch < self.skip_first_n_saves:
                msg = f"Skipping ckpt save at epoch {epoch} <= {self.skip_first_n_saves}"
                self.logger.info(msg)
                continue

            #if epoch % self.save_period == 0 or save_best:
            if save_best:
                self._save_checkpoint(epoch, log, save_best=best)


    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}"
                                ", but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, log, save_best=False):
        """Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'trained_model.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if self.include_optim_in_ckpts:
            state["optimizer"] = self.optimizer.state_dict()

        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        score_filename = str(self.checkpoint_dir / 'score-epoch{}.pt'.format(epoch))

        tic = time.time()
        #self.logger.info("Saving checkpoint: {} ...".format(filename))
        #torch.save(state, filename)
        #torch.save(log, score_filename)
        #self.logger.info(f"Done in {time.time() - tic:.3f}s")
        if save_best:
            self.logger.info("Updating 'best' checkpoint: {} ...".format(filename))
            best_path = str(self.checkpoint_dir / 'trained_model.pth')
            best_score_path = str(self.checkpoint_dir / 'best_score.pt')
            torch.save(state, best_path)
            torch.save(log, best_score_path)
            self.logger.info(f"Done in {time.time() - tic:.3f}s")
        else:
            torch.save(state, filename)
            torch.save(log, score_filename)

    def _resume_checkpoint(self, resume_path):
        """ Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        ckpt_path = self.config.resume / 'trained_model.pth'
        self.logger.info("Loading checkpoint: {} ...".format(ckpt_path))
        checkpoint = torch.load(ckpt_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            msg = ("Warning: Architecture configuration given in config file is"
                   "different from that of checkpoint. This may yield an exception"
                   " while state_dict is being loaded.")
            self.logger.warning(msg)
        
        self.model.load_state_dict(checkpoint['state_dict'])
        #for key in self.model.state_dict():
        #    self.model.state_dict()[key].copy_(checkpoint['state_dict'][key])
                    
        if self.include_optim_in_ckpts and self.optimizer is not None:
            # load optimizer state from ckpt only when optimizer type is not changed.
            optim_args = checkpoint['config']['optimizer']
            if optim_args['type'] != self.config['optimizer']['type']:
                msg = ("Warning: Optimizer type given in config file differs from that"
                       " of checkpoint. Optimizer parameters not being resumed.")
                self.logger.warning(msg)
            else:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger.info(f"Ckpt loaded. Resume training from epoch {self.start_epoch}")
