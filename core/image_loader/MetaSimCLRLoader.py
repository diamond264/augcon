# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import torchvision.transforms.functional as TF
import random
from torchvision.transforms import transforms
import core.image_loader.transforms as augs

class MetaSimCLRLoader:
    """Take two random crops of one image as the query and key."""
    def __init__(self, pre_transform, post_transform, cascade=False):
        self.pre_transform = pre_transform
        self.post_transform = post_transform

        size = self.pre_transform.transforms[0].size[0]
        
        random_crop = augs.get_random_crop(size = size)
        color_distortion = augs.get_color_distortion()
        gaussian_blur = augs.get_gaussian_blur(size = size)
        self.augmentations = transforms.Compose([
            random_crop,
            color_distortion,
            gaussian_blur
        ])

    def __call__(self, x):
        x = self.pre_transform(x)
        q = self.augmentations(x)
        q = self.post_transform(q)
        k = self.augmentations(x)
        k = self.post_transform(k)
        x = self.post_transform(x)
        
        return [x, q, k]