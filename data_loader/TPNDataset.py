import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from data_loader.transforms import *

NOISE_POS = 0
SCALE_POS = 1
ROTATE_POS = 2
NEGATE_POS = 3
FLIP_POS = 4
PERMUTE_POS = 5
TIME_WARP_POS = 6
CHANNEL_SHUFFLE_POS = 7

class TPNDataset(Dataset):
    def __init__(self, tensor_dataset):
        self.tensor_dataset = tensor_dataset
        
    def __len__(self):
        return len(self.tensor_dataset)
    
    def __getitem__(self, idx):
        features, classes, domains = self.tensor_dataset[idx]
        
        # new_features = features.numpy()
        tpn_labels = self.generate_labels()
        # new_features = torch.from_numpy(new_features).float()
        
        return features, features, tpn_labels, classes, domains
    
    def generate_labels(self):
        label = [0] * 8
        
        # # noise:
        # choice = np.random.choice(2, 1, p=[0.5, 0.5])[0]
        # x = tpn_noise(x, choice)
        # label[NOISE_POS] = choice
        
        # # scale:
        # choice = np.random.choice(2, 1, p=[0.5, 0.5])[0]
        # x = tpn_scale(x, choice)
        # label[SCALE_POS] = choice
        
        # # rotate:
        # choice = np.random.choice(2, 1, p=[0.5, 0.5])[0]
        # x = tpn_rotate(x, choice)
        # label[ROTATE_POS] = choice
        
        # # negate:
        # choice = np.random.choice(2, 1, p=[0.5, 0.5])[0]
        # x = tpn_negate(x, choice)
        # label[NEGATE_POS] = choice
        
        # # time_reversal:
        # choice = np.random.choice(2, 1, p=[0.5, 0.5])[0]
        # x = tpn_flip(x, choice)
        # label[FLIP_POS] = choice
        
        # # permutation:
        # choice = np.random.choice(2, 1, p=[0.5, 0.5])[0]
        # x = tpn_permute(x, choice)
        # label[PERMUTE_POS] = choice
        
        # # time_warped:
        # choice = np.random.choice(2, 1, p=[0.5, 0.5])[0]
        # x = tpn_time_warp(x, choice)
        # label[TIME_WARP_POS] = choice
        
        # # channel_shuffle:
        # choice = np.random.choice(2, 1, p=[0.5, 0.5])[0]
        # x = tpn_channel_shuffle(x, choice)
        # label[CHANNEL_SHUFFLE_POS] = choice
        
        return label