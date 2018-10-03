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

        for elem in ['image', 'real_mask', 'obs_mask']:
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
        for elem in ['image', 'real_mask', 'obs_mask']:
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


class PolygonMask(object):
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
        landmark = sample['landmark']
        gender = sample['gender']
        fake_gender = random.randint(1, 2)

        resolution = image.shape[-2]
        landmark_adjust_ratio = 256 // resolution
        real_mask = np.zeros([resolution, resolution],
                             dtype=np.uint8)
        obs_mask = real_mask.copy()

        polygon_coords = np.take(landmark, [0, 1, 2, 3, 8, 9, 6, 7])
        polygon_coords = polygon_coords.astype(np.int32).reshape(1, -1, 2)
        polygon_coords = polygon_coords // landmark_adjust_ratio

        cv2.fillPoly(real_mask, polygon_coords, int(gender))
        cv2.fillPoly(obs_mask, polygon_coords, fake_gender)

        assert len(image.shape) == 3, \
            f'image dims should be 3, not {len(image.shape)}'

        sample['real_mask'] = real_mask
        sample['obs_mask'] = obs_mask
        sample['fake_gender'] = fake_gender
        return sample

    def __str__(self):  # noqa: D105
        return f'PolygonMask:(num_classes={str(self.num_classes)})'
