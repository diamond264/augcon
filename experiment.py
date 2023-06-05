#!/usr/bin/env python

import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from util.args import parse_args
from util.config import Config
from util.logger import Logger

from core.CPC import CPCLearner
from core.MetaCPC import MetaCPCLearner
from data_loader.default_data_loader import DefaultDataLoader

class Experiment:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        
        self.configure_gpus()
        self.configure_random_seed()
    
    def configure_gpus(self):
        self.logger.info(f'Setting GPUs: {self.cfg.gpu}')
        self.gpu = []
        if self.cfg.gpu == 'all':
            self.gpu = [0,1,2,3,4,5,6,7]
        elif isinstance(self.cfg.gpu, int):
            self.gpu = [self.cfg.gpu]
        elif isinstance(self.cfg.gpu, list) and len(self.cfg.gpu) > 0:
            self.gpu = self.cfg.gpu
        else:
            self.logger.warning(f'Only GPU training is supported. Setting gpu 0 as default.')
            self.gpu = [0]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu) for gpu in self.gpu])
        self.logger.info(f'GPUs {self.gpu} will be used')
    
    def configure_random_seed(self):
        cudnn.benchmark = True
        if not self.cfg.seed is None:
            random.seed(self.cfg.seed)
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)
            cudnn.benchmark = False
            cudnn.deterministic = True
            torch.cuda.manual_seed_all(self.cfg.seed)
            
            self.logger.warning('You have chosen to seed training. ')
            self.logger.warning('This will turn on the CUDNN deterministic setting, ')
            self.logger.warning('which can slow down your training considerably! ')
            self.logger.warning('You may see unexpected behavior when restarting ')
            self.logger.warning('from checkpoints.')
    
    def run(self):
        # Model creation
        learner = None
        if self.cfg.pretext == 'cpc':
            learner = CPCLearner(self.cfg, self.gpu, self.logger)
        if self.cfg.pretext == 'metacpc':
            learner = MetaCPCLearner(self.cfg, self.gpu, self.logger)
        else:
            self.logger.warning('Pretext task not supported')
        
        # Loading dataset
        default_data_loader = DefaultDataLoader(self.cfg, self.logger)
        train_dataset, val_dataset, test_dataset = default_data_loader.get_datasets()
        
        # Start training
        learner.run(train_dataset, val_dataset, test_dataset)

if __name__ == '__main__':
    args = parse_args()
    
    cfg = Config(args.config)
    
    L = Logger()
    L.set_log_name(args.config)
    logger = L.get_logger()

    logger.info("================= Config =================")    
    cfg.log_config(logger)
    logger.info("==========================================")
    
    exp = Experiment(cfg, logger)
    exp.run()
