import random
import torch
import pickle

class DefaultDataLoader():
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        
        self.load_dataset()
    
    def load_dataset(self):
        if self.cfg.dataset_name == 'hhar':
            with open(self.cfg.train_dataset_path, 'rb') as f:
                self.train_dataset = pickle.load(f)
                indices = random.sample(range(len(self.train_dataset)), 15000)
                self.train_dataset = torch.utils.data.Subset(self.train_dataset, indices)
            with open(self.cfg.test_dataset_path, 'rb') as f:
                self.test_dataset = pickle.load(f)
            with open(self.cfg.val_dataset_path, 'rb') as f:
                self.val_dataset = pickle.load(f)
    
    def get_loaders(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            drop_last=True
        )
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            drop_last=True
        )
        
        return train_loader, val_loader, test_loader