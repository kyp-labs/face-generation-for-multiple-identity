"""Custom dataset classes."""

import os
import re
import glob

import numpy as np
import pandas as pd
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


class VGGFace2Dataset(Dataset):
    """VGGFace2 Dataset according to the resolution."""

    def __init__(self, data_dir, resolution, landmark_info_path,
                 identity_info_path, transform=None):
        """Constructor.

        Args:
            data_dir (str): Directory path containing dataset.
            resolution (int): Specific resolution value to load.
            landmark_info (str): Path of the file having landmark information.
            identity_info (str): Path of the file having identity information.
            transform: Augmentation options, Default is None.
                       (e.g. torchvision.transforms.Compose([
                                transform.CenterCrop(10),
                                transform.ToTensor(),
                                ]))
        """
        self.file_list = []
        dir_list = os.listdir(os.path.join(data_dir, str(resolution)))
        num_ids = len(dir_list)

        for ext in ('*.gif', '*.png', '*.jpg'):
            full_path = os.path.normcase(data_dir + f'/{resolution}/*/' + ext)
            self.file_list.extend(glob.glob(full_path))

        landmark_info = pd.read_csv(landmark_info_path)
        landmark_info = landmark_info[landmark_info['NAME_ID']
                                      .str.contains('|'.join(dir_list))]
        identity_info = pd.read_csv(identity_info_path)
        identity_info = identity_info[identity_info['Class_ID']
                                      .str.contains('|'.join(dir_list))]

        self.landmark_info = landmark_info
        self.identity_info = identity_info

        cls_ids = self.identity_info['Class_ID']
        self.id_to_identity = {i: j for i, j in zip(range(1, len(cls_ids)+1),
                                                    cls_ids)}
        self.identity_to_id = {j: i for i, j in self.id_to_identity.items()}
        self.transform = transform

    def __getitem__(self, idx):
        """Getter.

        Args:
            idx (int): index of image list.

        Return:
            sample (dict): {str: array} formatted data for training.
        """
        pattern = re.compile('n[0-9]{6}/[0-9]{4}_[0-9]{2}')
        image_path = self.file_list[idx]
        # For Windows OS
        image_path = image_path.replace("\\", "/")
        name_id = re.search(pattern, image_path)[0]

        identity = name_id.split('/')[0]
        id = self.identity_to_id[identity]

        landmark = self.landmark_info[self.landmark_info['NAME_ID'] ==
                                      name_id].iloc[:, 1:].values.flatten()
        image_arr = np.array(Image.open(image_path))

        sample = {'image': image_arr, 'landmark': landmark, 'id': id}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):  # noqa: D105
        return len(self.file_list)
