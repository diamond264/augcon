import torch
import random
from torch.utils.data import Dataset

from data_loader.transforms import *

class SimCLRDataset(Dataset):
    def __init__(self, tensor_dataset):
        self.tensor_dataset = tensor_dataset
        
    def __len__(self):
        return len(self.tensor_dataset)
    
    def __getitem__(self, idx):
        features, classes, domains = self.tensor_dataset[idx]
        augmented_features_1 = self.augment_features(features)
        augmented_features_2 = self.augment_features(features)
        
        augmented_features_1 = torch.from_numpy(augmented_features_1).float()
        augmented_features_2 = torch.from_numpy(augmented_features_2).float()
        # return augmented_features_1, augmented_features_2, classes, domains
        return features, augmented_features_1, augmented_features_2, classes, domains
    
    def augment_features(self, features):
        new_features = features.numpy()
        augmentations = [
            noise_transform_vectorized,
            scaling_transform_vectorized,
            rotation_transform_vectorized,
            negate_transform_vectorized,
            time_flip_transform_vectorized,
            channel_shuffle_transform_vectorized,
            time_segment_permutation_transform_vectorized,
            time_warp_transform_vectorized
        ]
        random.shuffle(augmentations)
        
        for aug in augmentations:
            new_features = aug(new_features)
            break
        
        return new_features.copy()