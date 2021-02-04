import os
import argparse
import time
import random
import numpy as np
import torch
from parse_config import ConfigParser
from utils import compute_dims
import data_loader.data_loaders as module_data
import model.model as module_arch
import model.loss as module_loss
import trainer as Trainer
from test import test

def main(config):
    logger = config.get_logger('train')
    expert_dims = compute_dims(config)
    seeds = [int(x) for x in config._args.seeds.split(',')]

    for seed in seeds:
        tic = time.time()
        logger.info(f"Setting experiment random seed to {seed}")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        data_loaders = config.init(
            name='data_loader',
            module=module_data,
            expert_dims=expert_dims,
            text_feat=config['experts']['text_feat'],
            text_dim=config['experts']['text_dim'],
        )

        model = config.init(
            name='arch',
            module=module_arch,
            expert_dims=expert_dims,
            text_dim=config['experts']['text_dim'],
            same_dim=config['experts']['ce_shared_dim'],
            text_feat=config['experts']['text_feat'],
            ref = config['trainer']['train_type'] in ['ref','refcomp','refjoint']
        )
        # logger.info(model)

        loss = config.init(name='loss', module=module_loss)

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init('lr_scheduler', torch.optim.lr_scheduler, optimizer)
        if config['trainer']['train_type'] in ['ref','refcomp','refjoint']:
            trainer = Trainer.TrainerRef(
                model,
                loss,
                optimizer,
                config=config,
                data_loaders=data_loaders,
                lr_scheduler=lr_scheduler,
            )
        else:
            trainer = Trainer.TrainerJoint(
                model,
                loss,
                optimizer,
                config=config,
                data_loaders=data_loaders,
                lr_scheduler=lr_scheduler,
            )
        

        trainer.train()
        best_ckpt_path = config.save_dir / "trained_model.pth"
        duration = time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - tic))
        logger.info(f"Training took {duration}")



if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--config', default='configs/ce/train.json', type=str)
    args.add_argument('--logdir', default='base', type=str)
    args.add_argument('--device', default=None, type=str)
    args.add_argument('--resume', default=None, type=str)
    args.add_argument('--seeds', default="0", type=str)
    args = ConfigParser(args, 'train')

    print("Launching experiment with config:")
    print(args)
    main(args)
