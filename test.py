import os
import argparse
from pathlib import Path
import time
import torch
from parse_config import ConfigParser
import data_loader.data_loaders as module_data
import model.model as module_arch
from model.metric import sharded_cross_view_inner_product
from utils.util import compute_dims
from utils.vocab import Vocabulary
from utils.text2vec import get_we_parameter
from trainer import TrainerJoint
import random
import numpy as np

def test(config):
    logger = config.get_logger('test')
    logger.info("Running test with configuration:")
    logger.info(config)

    expert_dims = compute_dims(config)

    vocab = None
    vocab_size = None
    we_parameter = None

    if "attr" in config['experts']['modalities']:
        attr_vocab = Vocabulary()
        attr_vocab.load(os.path.join(config['data_loader']['args']['data_dir'],'attributes/dict.attr.json'))
        attr_vocab_size = len(attr_vocab)
    else:
        attr_vocab = None
        attr_vocab_size = None
        
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
        text_feat=config['experts']['text_feat']
    )
    trainer = TrainerJoint(
        model,
        loss=None,
        optimizer=None,
        config=config,
        data_loaders=data_loaders,
        lr_scheduler=None,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running test on {device}")

    metric = trainer._valid_epoch(save_textatt = True)
    
    if config._args.mode == 'val':
        for key, value in metric.items():
            if key == 'recall_avg':
                logger.info(f'[Avg Recall]     : {value}')
            elif key == 'recall_avg_corr':
                logger.info(f'[Avg Recall corr]: {value}')
            elif key == 'comb_avg':
                logger.info(f'[comb_avg]       : {value}')
            elif key == 'recall':
                for i, category in zip(value, trainer.categories):
                    if len(i) == 2:
                        logger.info(f'[{category}] r@10, r@50: {i[0]}\t{i[1]}')
                    elif len(i) == 4:
                        logger.info(f'[{category}] comp corr r@10, r@50: {i[0]}\t{i[1]}\t{i[2]}\t{i[3]}')
            elif key == 'comb':
                combstr = "comb:"
                for i, category in zip(value, trainer.categories):
                    combstr += f' {i[0]} {i[1]}'
                logger.info(combstr)
    else:
        save_fname = config.save_dir / f'test_score.pt'
        tic = time.time()
        logger.info("Saving score matrix: {} ...".format(save_fname))
        torch.save(metric, save_fname)
        logger.info(f"Done in {time.time() - tic:.3f}s")


    
# python test.py --resume logdir/attr_noce

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--resume', default='', type=str, help='path to checkpoint for test')
    args.add_argument('--device', default='', type=str, help='indices of GPUs to enable')
    args.add_argument('--seeds', default="0", type=str)
    args.add_argument('--mode', default="val", type=str)
    
    test_config = ConfigParser(args, 'eval')

    seed = 0
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    msg = "For evaluation, a model checkpoint must be specified via the --resume flag"
    assert test_config._args.resume, msg
    #test_config.config['arch']['args']['score_module'] = 'inner_product_still'
    test(test_config)
