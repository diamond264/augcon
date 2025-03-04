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
from data_loader.DigitFiveDataset import DigitFiveDataset
from data_loader.PACSDataset import PACSDataset
from data_loader.Country211Dataset import Country211Dataset
from data_loader.PCamDataset import PCamDataset
from data_loader.ImageNetDataset import ImageNetDataset
from data_loader.EmptyDataset import EmptyDataset

# 1D pretext tasks
from core.CPC import CPCLearner
from core.SimCLR1D import SimCLR1DLearner

# 1D meta pretext tasks
from core.MetaCPC import MetaCPCLearner
from core.MetaSimCLR1D import MetaSimCLR1DLearner

# 2D pretext tasks
from core.SimCLR2D import SimCLR2DLearner
from core.SimSiam2D import SimSiam2DLearner

# 2D meta pretext tasks
from core.MetaSimSiam2D import MetaSimSiam2DLearner
from core.ReptileSimSiam2D import ReptileSimSiam2DLearner
from core.AdapterSimSiam2D import AdapterSimSiam2DLearner


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
            elif self.cfg.pretext == 'simclr':
                learner = SimCLR1DLearner(self.cfg, self.gpu, self.logger)
            else:
                self.logger.warning('Pretext task not supported')
        # For 2D data (images)
        elif self.cfg.dtype == '2d':
            if self.cfg.pretext == 'simclr':
                learner = SimCLR2DLearner(self.cfg, self.gpu, self.logger)
            elif self.cfg.pretext == 'simsiam':
                learner = SimSiam2DLearner(self.cfg, self.gpu, self.logger)
            elif self.cfg.pretext == 'metasimsiam':
                learner = ReptileSimSiam2DLearner(self.cfg, self.gpu, self.logger)
            elif self.cfg.pretext == 'adaptersimsiam':
                learner = AdapterSimSiam2DLearner(self.cfg, self.gpu, self.logger)
            else:
                self.logger.warning('Pretext task not supported')
        
        # Loading dataset
        if self.cfg.mode == 'finetune': episodes = self.cfg.episodes
        else: episodes = 1
        for episode in range(episodes):
            self.logger.info(f'================= Episode {episode} =================')
            if self.cfg.dtype == '1d':
                default_data_loader = DefaultDataLoader(self.cfg, self.logger)
                train_dataset, val_dataset, test_dataset = default_data_loader.get_datasets()
            elif self.cfg.dtype == '2d':
                if self.cfg.dataset_name == 'domainnet':
                    train_dataset = DomainNetDataset(self.cfg, self.logger, type='train')
                    label_dict = train_dataset.get_label_dict()
                    val_dataset = DomainNetDataset(self.cfg, self.logger, type='val', label_dict=label_dict)
                    test_dataset = DomainNetDataset(self.cfg, self.logger, type='test', label_dict=label_dict)
                elif self.cfg.dataset_name == 'digit5':
                    #fixme: change to digit5
                    train_dataset = DigitFiveDataset(self.cfg, self.logger, type='train')
                    label_dict = train_dataset.get_label_dict()
                    val_dataset = DigitFiveDataset(self.cfg, self.logger, type='val', label_dict=label_dict)
                    test_dataset = DigitFiveDataset(self.cfg, self.logger, type='test', label_dict=label_dict)
                elif self.cfg.dataset_name == 'pacs':
                    train_dataset = PACSDataset(self.cfg, self.logger, type='train')
                    label_dict = train_dataset.get_label_dict()
                    val_dataset = PACSDataset(self.cfg, self.logger, type='val', label_dict=label_dict)
                    test_dataset = PACSDataset(self.cfg, self.logger, type='test', label_dict=label_dict)
                #todo: Add Office-Home

                elif self.cfg.dataset_name == 'country211':
                    train_dataset = Country211Dataset(self.cfg, self.logger, type='train')
                    label_dict = train_dataset.get_label_dict()
                    val_dataset = Country211Dataset(self.cfg, self.logger, type='val', label_dict=label_dict)
                    test_dataset = Country211Dataset(self.cfg, self.logger, type='test', label_dict=label_dict)
                # todo: Add PCam
                elif self.cfg.dataset_name == 'pcam':
                    train_dataset = PCamDataset(self.cfg, self.logger, type='train')
                    label_dict = train_dataset.get_label_dict()
                    val_dataset = PCamDataset(self.cfg, self.logger, type='val', label_dict=label_dict)
                    test_dataset = PCamDataset(self.cfg, self.logger, type='test', label_dict=label_dict)
                elif 'imagenet' in self.cfg.dataset_name: # both for imagenet & imagenet_mini
                    train_dataset = ImageNetDataset(self.cfg, self.logger, type='train')
                    label_dict = train_dataset.get_label_dict()
                    val_dataset = ImageNetDataset(self.cfg, self.logger, type='val', label_dict=label_dict)
                    test_dataset = ImageNetDataset(self.cfg, self.logger, type='test', label_dict=label_dict)

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
