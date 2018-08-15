"""Custom dataset classes."""

import os
import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CelebADataset(Dataset):
    """CelebA Dataset according to the resolution."""

    def __init__(self, data_dir, resolution, transform=None):
        """Constructor.

        Args:
            data_dir (str): Directory path containing dataset.
            resolution (int): Specific resolution value to load.
            transform: Augmentation options, Default is None.
                       (e.g. torchvision.transforms.Compose([
                                transform.CenterCrop(10),
                                transform.ToTensor(),
                                ]))
        """
        self.file_list = glob.glob(data_dir + f'{resolution}/*.jpg')
        self.transform = transform
        self.resolution = resolution

    def __getitem__(self, idx):
        """Getter.

        Args:
            idx (int): index of image list.

        Return:
            sample (dict): {str: array} formatted data for training.
        """
        image_path = self.file_list[idx]
        image_arr = np.array(Image.open(image_path))
        attr_name = os.path.basename(image_path).split('_')[0]

        sample = {'image': image_arr, 'attr': attr_name}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):  # noqa: D105
        return len(self.file_list)


class CelebAHQDataset(Dataset):
    """CelebA-HQ Dataset according to the resolution."""

    def __init__(self, data_dir, resolution, transform=None):
        """Constructor.

        Args:
            data_dir (str): Directory path containing dataset.
            resolution (int): Specific resolution value to load.
            transform: Augmentation options, Default is None.
                       (e.g. torchvision.transforms.Compose([
                                transform.CenterCrop(10),
                                transform.ToTensor(),
                                ]))
        """
        self.file_list = glob.glob(data_dir + f'{resolution}/*.png')
        self.transform = transform
        self.resolution = resolution

    def __getitem__(self, idx):
        """Getter.

        Args:
            idx (int): index of image list.

        Return:
            sample (dict): {str: array} formatted data for training.
        """
        image_path = self.file_list[idx]
        # remove transparency channel
        image_arr = np.array(Image.open(image_path))[:, :, :3]
        attr_str = os.path.basename(image_path).split('_')[0]
        attr_arr = np.array([int(i) for i in attr_str])

        sample = {'image': image_arr, 'attr': attr_arr}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):  # noqa: D105
        return len(self.file_list)
