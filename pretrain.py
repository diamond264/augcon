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
from data_loader.default_data_loader import DefaultDataLoader

class Pretrain:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        
        self.configure_gpus()
        self.configure_random_seed()
    
    def configure_gpus(self):
        self.logger.info(f'Setting visible GPUs: {self.cfg.visible_gpus}')
        if self.cfg.visible_gpus == 'all':
            self.cfg.visible_gpus = '0,1,2,3,4,5,6,7'
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cfg.visible_gpus
        visible_gpus = [int(gpu) for gpu in self.cfg.visible_gpus.split(',')]
        
        self.gpu = self.cfg.gpu
        if self.gpu is None:
            self.logger.warning(f'Only GPU training is supported. Setting gpu 0 as default.')
            self.gpu = [visible_gpus[0]]
        if self.gpu == 'all':
            self.gpu = visible_gpus
        else:
            if isinstance(self.gpu, int):
                self.gpu = [visible_gpus[self.gpu]]
            else:
                self.gpu = [visible_gpus[int(idx)] for idx in self.gpu.split(',')]
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
        # if self.cfg.pretext == 'metacpc':
        #     learner = MetaCPCLearner(self.cfg, self.gpu, self.logger)
        else:
            self.logger.warning('Pretext task not supported')
        
        # Criterion
        if self.cfg.criterion == 'crossentropy':
            criterion = nn.CrossEntropyLoss().cuda(self.gpu[0])
        else:
            self.logger.warning('Criterion not supported')
        
        # Optimizer
        parameters = learner.parameters()
        if self.cfg.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                parameters, self.cfg.lr,
                momentum=self.cfg.momentum,
                weight_decay=self.cfg.wd
            )
        elif self.cfg.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                parameters, self.cfg.lr, 
                weight_decay=self.cfg.wd
            )
        
        # Resume previous training
        if os.path.isfile(self.cfg.resume):
            state = torch.load(self.cfg.resume)
            learner.load_state_dict(state['state_dict'])
            optimizer.load_state_dict(state['optimizer'])
            self.cfg.start_epoch = state['epoch']
        
        # Loading dataset
        default_data_loader = DefaultDataLoader(self.cfg, self.logger)
        train_loader, val_loader, test_loader = default_data_loader.get_loaders()
        
        # Start training
        learner.perform_train(train_loader, val_loader, test_loader, criterion, optimizer)

if __name__ == '__main__':
    args = parse_args()
    
    cfg = Config(args.config)
    
    L = Logger()
    L.set_log_name(args.config)
    logger = L.get_logger()

    logger.info("================= Config =================")    
    cfg.log_config(logger)
    logger.info("==========================================")
    
    pretrain = Pretrain(cfg, logger)
    pretrain.run()
