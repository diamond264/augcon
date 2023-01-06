# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import random
import numpy as np
import torch
from torchvision.transforms.transforms import Compose

random_mirror = True


def ShearX(original_img, scale=None, range_=[-0.3, 0.3]):
    img = original_img.copy()

    if not scale is None:
        if scale < range_[0] or scale > range_[1]:
            assert(0)
    else:
        level = random.random()
        scale = (range_[1]-range_[0])*level+range_[0]
    
    return img.transform(img.size, PIL.Image.AFFINE, (1, scale, 0, 0, 1, 0)), scale