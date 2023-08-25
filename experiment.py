#!/usr/bin/env python

import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset

from util.args import parse_args
from util.config import Config
from util.logger import Logger

from data_loader.default_data_loader import DefaultDataLoader
from data_loader.DomainNetDataset import DomainNetDataset
from data_loader.EmptyDataset import EmptyDataset

from core.CPC import CPCLearner
# from core.SimCLR import SimCLRLearner

from core.MetaCPC import MetaCPCLearner
from core.MetaSimCLR import MetaSimCLRLearner
from core.ReptileSimSiam2D import ReptileSimSiamLearner

from core.SimCLR2D import SimCLRLearner
from core.SimSiam2D import SimSiamLearner

from core.MetaSimSiam2D import MetaSimSiamLearner


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
        
        # For 1D data (sensors, audio)
        if self.cfg.dtype == '1d':
            if self.cfg.pretext == 'cpc':
                learner = CPCLearner(self.cfg, self.gpu, self.logger)
            elif self.cfg.pretext == 'metacpc':
                learner = MetaCPCLearner(self.cfg, self.gpu, self.logger)
            # elif self.cfg.pretext == 'simclr' and self.cfg.dtype=='1d':
            #     learner = SimCLRLearner(self.cfg, self.gpu, self.logger)
            else:
                self.logger.warning('Pretext task not supported')
        # For 2D data (images)
        elif self.cfg.dtype == '2d':
            if self.cfg.pretext == 'simclr':
                learner = SimCLRLearner(self.cfg, self.gpu, self.logger)
            elif self.cfg.pretext == 'simsiam':
                learner = SimSiamLearner(self.cfg, self.gpu, self.logger)
            elif self.cfg.pretext == 'metasimclr':
                learner = MetaSimCLRLearner(self.cfg, self.gpu, self.logger)
            elif self.cfg.pretext == 'metasimsiam':
                # learner = MetaSimSiamLearner(self.cfg, self.gpu, self.logger)
                learner = ReptileSimSiamLearner(self.cfg, self.gpu, self.logger)
            else:
                self.logger.warning('Pretext task not supported')
        
        # Loading dataset
        if self.cfg.mode == 'finetune': episodes = self.cfg.episodes
        else: episodes = 1
        for episode in range(episodes):
            if self.cfg.dtype == '1d':
                default_data_loader = DefaultDataLoader(self.cfg, self.logger)
                train_dataset, val_dataset, test_dataset = default_data_loader.get_datasets()
            elif self.cfg.dtype == '2d':
                if self.cfg.dataset_name == 'domainnet':
                    train_dataset = DomainNetDataset(self.cfg, self.logger, self.cfg.train_dataset_path, type='train')
                    label_dict = train_dataset.get_label_dict()
                    val_dataset = DomainNetDataset(self.cfg, self.logger, self.cfg.val_dataset_path, type='val', label_dict=label_dict)
                    test_dataset = DomainNetDataset(self.cfg, self.logger, self.cfg.test_dataset_path, type='test', label_dict=label_dict)
            
            # For debugging
            # if len(train_dataset) > 180000:
            #     train_dataset = Subset(train_dataset, np.random.choice(len(train_dataset), 180000, replace=False))
            
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
