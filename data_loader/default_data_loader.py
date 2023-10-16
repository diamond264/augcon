import random
import torch
import pickle
from data_loader.CPCDataset import CPCDataset
from data_loader.SimCLRDataset import SimCLRDataset
from data_loader.TPNDataset import TPNDataset

class DefaultDataLoader():
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.load_dataset()
    
    def load_dataset(self):
        with open(self.cfg.train_dataset_path, 'rb') as f:
            self.train_dataset = pickle.load(f)
            # if len(self.train_dataset) > 15000:
            #     indices = random.sample(range(len(self.train_dataset)), 15000)
            #     self.train_dataset = torch.utils.data.Subset(self.train_dataset, indices)
        with open(self.cfg.test_dataset_path, 'rb') as f:
            self.test_dataset = pickle.load(f)
        with open(self.cfg.val_dataset_path, 'rb') as f:
            self.val_dataset = pickle.load(f)
    
    def get_datasets(self):
        if self.cfg.pretext == 'tpn':
            self.train_dataset = TPNDataset(self.train_dataset)
            self.val_dataset = TPNDataset(self.val_dataset)
            self.test_dataset = TPNDataset(self.test_dataset)
        if self.cfg.pretext == 'cpc' or \
            self.cfg.pretext == 'metacpc' or\
            self.cfg.pretext == 'autoencoder':
            self.train_dataset = CPCDataset(self.train_dataset)
            self.val_dataset = CPCDataset(self.val_dataset)
            self.test_dataset = CPCDataset(self.test_dataset)
        if self.cfg.pretext == 'simclr' or \
            self.cfg.pretext == 'metasimclr' or \
            self.cfg.pretext == 'simsiam' or \
            self.cfg.pretext == 'metasimsiam':
            self.train_dataset = SimCLRDataset(self.train_dataset)
            self.val_dataset = SimCLRDataset(self.val_dataset)
            self.test_dataset = SimCLRDataset(self.test_dataset)
        
        return self.train_dataset, self.val_dataset, self.test_dataset