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

# from data_loader.DomainNetDataset import DomainNetDataset
# from data_loader.DigitFiveDataset import DigitFiveDataset
# from data_loader.PACSDataset import PACSDataset
# from data_loader.Country211Dataset import Country211Dataset
# from data_loader.PCamDataset import PCamDataset
# from data_loader.ImageNetDataset import ImageNetDataset
# from data_loader.EmptyDataset import EmptyDataset

# 1D pretext tasks
from core.CPC import CPCLearner
from core.SimCLR1D import SimCLR1DLearner
from core.SimSiam1D import SimSiam1DLearner
from core.AutoEncoder import AutoEncoderLearner
from core.TPN import TPNLearner

# 1D meta pretext tasks
from core.MetaCPC import MetaCPCLearner
from core.MetaSimCLR1D import MetaSimCLR1DLearner
from core.MetaSimSiam1D import MetaSimSiam1DLearner
from core.MetaTPN import MetaTPNLearner
from core.MetaAutoEncoder import MetaAutoEncoderLearner


class Experiment:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

        self.configure_gpus()
        self.configure_random_seed()

    def configure_gpus(self):
        self.logger.info(f"Setting GPUs: {self.cfg.gpu}")
        self.gpu = []
        if self.cfg.gpu == "all":
            self.gpu = [0, 1, 2, 3, 4, 5, 6, 7]
        elif isinstance(self.cfg.gpu, int):
            self.gpu = [self.cfg.gpu]
        elif isinstance(self.cfg.gpu, list) and len(self.cfg.gpu) > 0:
            self.gpu = self.cfg.gpu
        else:
            self.logger.warning(
                f"Only GPU training is supported. Setting gpu 0 as default."
            )
            self.gpu = [0]
        # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu) for gpu in self.gpu])
        self.logger.info(f"GPUs {self.gpu} will be used")

    def configure_random_seed(self):
        pass

    def run(self):
        # Model creation
        learner = None

        # For 1D data (sensors, audio)
        if self.cfg.pretext == "cpc":
            learner = CPCLearner(self.cfg, self.gpu, self.logger)
        elif self.cfg.pretext == "simclr":
            learner = SimCLR1DLearner(self.cfg, self.gpu, self.logger)
        elif self.cfg.pretext == "autoencoder":
            learner = AutoEncoderLearner(self.cfg, self.gpu, self.logger)
        elif self.cfg.pretext == "tpn":
            learner = TPNLearner(self.cfg, self.gpu, self.logger)
        elif self.cfg.pretext == "simsiam":
            learner = SimSiam1DLearner(self.cfg, self.gpu, self.logger)
        # Meta learning methods
        elif self.cfg.pretext == "metacpc":
            learner = MetaCPCLearner(self.cfg, self.gpu, self.logger)
        elif self.cfg.pretext in ["metasimclr", "setsimclr"]:
            learner = MetaSimCLR1DLearner(self.cfg, self.gpu, self.logger)
        elif self.cfg.pretext == "metatpn":
            learner = MetaTPNLearner(self.cfg, self.gpu, self.logger)
        elif self.cfg.pretext == "metaautoencoder":
            learner = MetaAutoEncoderLearner(self.cfg, self.gpu, self.logger)
        elif self.cfg.pretext == "metasimsiam":
            learner = MetaSimSiam1DLearner(self.cfg, self.gpu, self.logger)
        else:
            self.logger.warning("Pretext task not supported")

        default_data_loader = DefaultDataLoader(self.cfg, self.logger)
        train_dataset, val_dataset, test_dataset = default_data_loader.get_datasets()

        # Start training
        learner.run(train_dataset, val_dataset, test_dataset)  # , neg_dataset)


if __name__ == "__main__":
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
