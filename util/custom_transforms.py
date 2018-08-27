"""Custom transforms for augmentation."""

import random

import cv2
import numpy as np
import torch

"""
TODO: Scaling/Rotation makes values lower than -1 check
"""


class ScaleNRotate(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth."""

    def __init__(self, rots=(-30, 30), scales=(.75, 1.25)):
        """constructor.

        Args:
            rots (tuple): (minimum, maximum) rotation angle.
            scales (tuple): (minimum, maximum) scale.
        """
        assert (isinstance(rots, type(scales)))
        self.rots = rots
        self.scales = scales

    def __call__(self, sample):
        """caller.

        Args:
            sample (dict): {str: array} formatted data for training.

        Returns:
            sample (dict): {str: array} formatted data for training.

        """
        rot = (self.rots[1] - self.rots[0]) * random.random() - \
              (self.rots[1] - self.rots[0])/2

        sc = (self.scales[1] - self.scales[0]) * random.random() - \
             (self.scales[1] - self.scales[0]) / 2 + 1

        for elem in ['image', 'mask']:
            image = sample[elem]

            h, w = image.shape[:2]
            center = (w / 2, h / 2)
            assert(center != 0)  # Strange behavior warpAffine
            M = cv2.getRotationMatrix2D(center, rot, sc)

            if ((image == 0) | (image == 1)).all():
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC
            tmp = cv2.warpAffine(image, M, (w, h), flags=flagval)
            if image.ndim == 2:
                image = image[:, :, np.newaxis]

            sample[elem] = tmp
        return sample

    def __str__(self):  # noqa: D105
        return f'ScaleNRotate:(rot={str(self.rots)},scale={str(self.scales)})'


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        """caller.

        Args:
            sample (dict): {str: array} formatted data for training.

        Returns:
            sample (dict): {str: array} formatted data for training.

        """
        for elem in ['image', 'mask']:
            tmp = sample[elem]

            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            tmp = tmp.transpose((2, 0, 1))
            sample[elem] = torch.from_numpy(tmp).float()

        return sample


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation."""

    def __init__(self, mean, std):
        """constructor.

        Args:
            mean (float): mean value of an image.
            std (float): standard deviation value of an image.
        """
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """caller.

        Args:
            sample (dict): {str: array} formatted data for training.

        Returns:
            sample (dict): {str: array} formatted data for training.

        """
        for elem in ['image']:
            tmp = sample[elem]
            if tmp.max() > 1:
                tmp = tmp / 255.
            tmp = (tmp - self.mean) / self.std
            sample[elem] = tmp
        return sample

    def __str__(self):  # noqa: D105
        return self.__class__.__name__ + '(mean={0}, std={1})'.\
                format(self.mean, self.std)


class TargetMask(object):
    """Add Square mask to the sample."""

    def __init__(self, num_classes=10):
        """constructor."""
        self.num_classes = num_classes

    def __call__(self, sample):
        """caller.

        Args:
            sample (dict): {str: array} formatted data for training.

        Returns:
            sample (dict): {str: array} formatted data for training.

        """
        image = sample['image']
        target_id = random.randint(0, self.num_classes-1)

        resolution = image.shape[-2]
        mask = np.zeros([resolution, resolution])

        start_pos = resolution // 4
        end_pos = resolution * 3 // 4

        assert len(image.shape) == 3, \
            f'image dims should be 3, not {len(image.shape)}'

        mask[start_pos: end_pos, start_pos: end_pos] = target_id

        sample['mask'] = mask
        sample['target_id'] = target_id
        return sample

    def __str__(self):  # noqa: D105
        return f'TargetMask:(num_classes={str(self.num_classes)})'
