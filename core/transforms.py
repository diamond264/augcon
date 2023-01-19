# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import random
import numpy as np
import torch
from torchvision.transforms.transforms import Compose

random_mirror = True


def ShearX(original_img, range_=[-0.3, 0.3], prob=0.2, scale=None):
    img = original_img.copy()
    if random.random() >= prob:
        return img, 0, False

    if not scale is None:
        if scale < range_[0] or scale > range_[1]:
            assert(0)
    else:
        level = random.random()
        scale = (range_[1]-range_[0])*level+range_[0]
    
    return img.transform(img.size, PIL.Image.AFFINE, (1, scale, 0, 0, 1, 0)), scale, True


def ShearY(original_img, range_=[-0.3, 0.3], prob=0.2, scale=None):
    img = original_img.copy()
    if random.random() >= prob:
        return img, 0, False

    if not scale is None:
        if scale < range_[0] or scale > range_[1]:
            assert(0)
    else:
        level = random.random()
        scale = (range_[1]-range_[0])*level+range_[0]

    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, scale, 1, 0)), scale, True


def TranslateX(original_img, range_=[-0.45, 0.45], prob=0.2, scale=None):
    img = original_img.copy()
    if random.random() >= prob:
        return img, 0, False

    if not scale is None:
        if scale < range_[0] or scale > range_[1]:
            assert(0)
    else:
        level = random.random()
        scale = (range_[1]-range_[0])*level+range_[0]

    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, scale, 0, 1, 0)), scale, True


def TranslateY(original_img, range_=[-0.45, 0.45], prob=0.2, scale=None):
    img = original_img.copy()
    if random.random() >= prob:
        return img, 0, False

    if not scale is None:
        if scale < range_[0] or scale > range_[1]:
            assert(0)
    else:
        level = random.random()
        scale = (range_[1]-range_[0])*level+range_[0]

    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, scale)), scale, True


def Rotate(original_img, range_=[-90, 90], prob=0.2, scale=None):
    img = original_img.copy()
    if random.random() >= prob:
        return img, 0, False

    if not scale is None:
        if scale < range_[0] or scale > range_[1]:
            assert(0)
    else:
        level = random.random()
        scale = (range_[1]-range_[0])*level+range_[0]

    return img.rotate(scale), scale, True


def AutoContrast(original_img, range_=[0, 0], prob=0.2, scale=None):
    img = original_img.copy()
    if random.random() >= prob:
        return img, 0, False

    return PIL.ImageOps.autocontrast(img), 0, True


def Invert(original_img, range_=[0, 0], prob=0.2, scale=None):
    img = original_img.copy()
    if random.random() >= prob:
        return img, 0, False

    return PIL.ImageOps.invert(img), 0, True


def Equalize(original_img, range_=[0, 0], prob=0.2, scale=None):
    img = original_img.copy()
    if random.random() >= prob:
        return img, 0, False

    return PIL.ImageOps.equalize(img), 0, True


def HorizontalFlip(original_img, range_=[0, 0], prob=0.2, scale=None):
    img = original_img.copy()
    if random.random() >= prob:
        return img, 0, False

    return PIL.ImageOps.mirror(img), 0, True


def VerticalFlip(original_img, range_=[0, 0], prob=0.2, scale=None):
    img = original_img.copy()
    if random.random() >= prob:
        return img, 0, False

    return PIL.ImageOps.flip(img), 0, True
    

def Solarize(original_img, range_=[0, 256], prob=0.2, scale=None):
    img = original_img.copy()
    if random.random() >= prob:
        return img, 0, False

    if not scale is None:
        if scale < range_[0] or scale > range_[1]:
            assert(0)
    else:
        level = random.random()
        scale = (range_[1]-range_[0])*level+range_[0]

    return PIL.ImageOps.solarize(img, scale), scale, True


def Posterize(original_img, range_=[0, 8], prob=0.2, scale=None):
    img = original_img.copy()
    if random.random() >= prob:
        return img, 0, False

    if not scale is None:
        if scale < range_[0] or scale > range_[1]:
            assert(0)
    else:
        level = random.random()
        scale = (range_[1]-range_[0])*level+range_[0]

    scale = int(scale)
    if scale > 7: scale = 7

    return PIL.ImageOps.posterize(img, scale), scale, True


def Contrast(original_img, range_=[0.1, 1.9], prob=0.2, scale=None):
    img = original_img.copy()
    if random.random() >= prob:
        return img, 0, False

    if not scale is None:
        if scale < range_[0] or scale > range_[1]:
            assert(0)
    else:
        level = random.random()
        scale = (range_[1]-range_[0])*level+range_[0]

    return PIL.ImageEnhance.Contrast(img).enhance(scale), scale, True


def Color(original_img, range_=[0.1, 1.9], prob=0.2, scale=None):
    img = original_img.copy()
    if random.random() >= prob:
        return img, 0, False

    if not scale is None:
        if scale < range_[0] or scale > range_[1]:
            assert(0)
    else:
        level = random.random()
        scale = (range_[1]-range_[0])*level+range_[0]

    return PIL.ImageEnhance.Color(img).enhance(scale), scale, True


def Brightness(original_img, range_=[0.1, 1.9], prob=0.2, scale=None):
    img = original_img.copy()
    if random.random() >= prob:
        return img, 0, False

    if not scale is None:
        if scale < range_[0] or scale > range_[1]:
            assert(0)
    else:
        level = random.random()
        scale = (range_[1]-range_[0])*level+range_[0]

    return PIL.ImageEnhance.Brightness(img).enhance(scale), scale, True


def Sharpness(original_img, range_=[0.1, 1.9], prob=0.2, scale=None):
    img = original_img.copy()
    if random.random() >= prob:
        return img, 0, False

    if not scale is None:
        if scale < range_[0] or scale > range_[1]:
            assert(0)
    else:
        level = random.random()
        scale = (range_[1]-range_[0])*level+range_[0]

    return PIL.ImageEnhance.Sharpness(img).enhance(scale), scale, True


def Cutout(original_img, range_=[0, 0.2], prob=0.2, scale=None):
    img = original_img.copy()
    if random.random() >= prob:
        return img, 0, False

    if not scale is None:
        if scale < range_[0] or scale > range_[1]:
            assert(0)
    else:
        level = random.random()
        scale = (range_[1]-range_[0])*level+range_[0]

    scale_ = scale * img.size[0]

    return CutoutAbs(img, scale_), scale, True


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)
    
    vx = np.random.uniform(v)
    vy = np.random.uniform(v)

    x0 = int(max(0, x0 - vx / 2.))
    y0 = int(max(0, y0 - vy / 2.))
    x1 = min(w, x0 + vx / 2.)
    y1 = min(h, y0 + vy / 2.)
    
    color = (0, 0, 0)
    img = img.copy()
    
    xy = (x0, 0, x1, h)
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    xy = (0, y0, w, y1)
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


##### Newly added augmentations
def Hue(img, _):
    np_image = np.asarray(img)
    axis = np.random.uniform(low=-1, high=1, size=np_image.shape[2])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    np_image = np.matmul(np_image , axangle2mat(axis,angle)**2)
    np_image = np.clip(np_image, 0, 255)
    reorg_img = PIL.Image.fromarray(np_image.astype(np.uint8))
    return reorg_img

def rgb_to_hsv(rgb):
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv

def hsv_to_rgb(hsv):
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')


def none(img, v):
    return img