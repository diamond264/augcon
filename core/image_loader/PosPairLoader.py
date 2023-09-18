# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torchvision.transforms import transforms
import core.image_loader.transforms as augs
import random
from PIL import ImageFilter


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class PosPairLoader:
    """Take two random crops of one image as the query and key."""
    def __init__(self, pre_transform, post_transform, cascade=False, rand_aug=False, return_original=False):
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.return_original = return_original

        size = self.pre_transform.transforms[0].size[0]
        
        # if rand_aug:
        #     crop_prob = random.uniform(0.5, 1)
        #     color_prob = random.uniform(0.5, 1)
        #     blur_prob = random.uniform(0.5, 1)
        #     crop_s = random.uniform(1, 2)
        #     color_s = random.uniform(1, 2)
        #     blur_s = random.uniform(0.8, 1.2)
        # else:
        #     crop_prob = 1
        #     color_prob = 1
        #     blur_prob = 1
        #     crop_s = 1
        #     color_s = 1
        #     blur_s = 1
        
        # random_crop = augs.get_random_crop(size, crop_s, crop_prob)
        # color_distortion = augs.get_color_distortion(color_s, color_prob)
        # gaussian_blur = augs.get_gaussian_blur(size, blur_s, blur_prob)
        # self.augmentations = transforms.Compose([
        #     random_crop,
        #     color_distortion,
        #     gaussian_blur
        # ])
        self.augmentations = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip()
        ])

    def __call__(self, x):
        # x = self.pre_transform(x)
        
        q = self.augmentations(x)
        k = self.augmentations(x)
        
        q = self.post_transform(q)
        k = self.post_transform(k)
        
        if self.return_original:
            x = self.pre_transform(x)
            x = self.post_transform(x)
            return [x, q, k]
        else:
            return [q, k]