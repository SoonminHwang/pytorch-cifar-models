from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def drop_channel(img, ch, num_output_channels):
    """Drop a channel
    Args:
        img (PIL Image): PIL Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns:
        PIL Image: Hue adjusted image.
    """
    if num_output_channels != 1 and num_output_channels != 3:
        raise ValueError('num_output_channels should be in \{1, 3\}.')

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        l = np.array(img, dtype=np.uint8)
        l[...] = 0
        return Image.fromarray(l, 'L').convert(input_mode)

    np_img = np.array(img, dtype=np.uint8)
    np_img[:,:,ch] = 0 

    return Image.fromarray(np_img, input_mode)



class RandomChannelDrop(object):
    """Randomly drop R or B channel with a probability of p2.
    Args:
        p1 (float): probability that image should be remained the same.
        p2 (float): probability that R is dropped. (otherwise, drop B)
    Returns:
        PIL Image: Dropped version of the input image with probability (1-p1) 
        and unchanged with probability p1.
        - If input image is 1 channel: drop the channel or not
        - If input image is 3 channel: drop 1st or 3rd channel or not
    """

    def __init__(self, p1=0.5, p2=0.5):
        self.p1 = p1
        self.p2 = p2

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be considered channel drop.
        Returns:
            PIL Image: Randomly channel-dropped image.
        """
        num_output_channels = 1 if img.mode == 'L' else 3
        if random.random() < self.p1:
            if random.random() < self.p2:
                return drop_channel(img, 0, num_output_channels=num_output_channels)
            else:
                return drop_channel(img, 2, num_output_channels=num_output_channels)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p1={0}, p2={1})'.format(self.p1, self.p2)
