import torch
from torch.utils.data import Dataset

class CPCDataset(Dataset):
    def __init__(self, tensor_dataset):
        self.tensor_dataset = tensor_dataset
        
    def __len__(self):
        return len(self.tensor_dataset)
    
    def __getitem__(self, idx):
        features, classes, domains = self.tensor_dataset[idx]
        return features, classes, domains
