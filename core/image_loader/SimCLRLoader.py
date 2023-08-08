# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import torchvision.transforms.functional as TF
import random
import core.image_loader.transforms as transforms

class SimCLRLoader:
    """Take two random crops of one image as the query and key."""
    def __init__(self, pre_transform, post_transform, augmentations, cascade=False):
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.augmentations = augmentations
        self.cascade = cascade

    def __call__(self, x):
        x = self.pre_transform(x)
        
        q = transforms.apply(x, self.augmentations, cascade=self.cascade)
        k = transforms.apply(x, self.augmentations, cascade=self.cascade)
        
        q = self.post_transform(q)
        k = self.post_transform(k)
        
        return [q, k]