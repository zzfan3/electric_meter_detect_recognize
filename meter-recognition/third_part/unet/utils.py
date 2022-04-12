# -*- coding: utf-8 -*-
import torch
import numpy as np
import torchvision.transforms.functional as F


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs):
        for t in self.transforms:
            imgs = t(imgs)
        return imgs

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class SegToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pics):
        """
        Args:
            pics (PIL Image or numpy.ndarray): Image to be converted to tensor.
            pics = [RGB, GroundTruth]

        Returns:
            Tensor: Converted image.
        """
        results = []
        if isinstance(pics, list):
            for i, pic in enumerate(pics):
                if i == 0:  # For RGB
                    img = np.array(pic).astype(np.float32)
                    if len(img.shape) == 3:
                        pass
                    else:
                        img = img[:, :, np.newaxis]
                    mean = np.mean(img[img[..., 0] > 0], axis=0)
                    std = np.std(img[img[..., 0] > 0], axis=0)
                    pic = (img - mean) / (std + 1e-6)
                    img = torch.from_numpy(pic.transpose((2, 0, 1)))  # RGB / 灰度图
                    results.append(img)
                else:  # For GroundTruth
                    results.append(F.to_tensor(pic))
        else:
            img = np.array(pics).astype(np.float32)
            if len(img.shape) == 3:
                pass
            else:
                img = img[:, :, np.newaxis]
            mean = np.mean(img[img[..., 0] > 0], axis=0)
            std = np.std(img[img[..., 0] > 0], axis=0)
            pic = (img - mean) / (std + 1e-6)
            img = torch.from_numpy(pic.transpose((2, 0, 1)))  # RGB / 灰度图
            results.append(img)
        if len(results) == 1:
            results = results[0]
        return results

    def __repr__(self):
        return self.__class__.__name__ + '()'
