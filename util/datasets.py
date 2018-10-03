"""Custom dataset classes.

Directory structure:
    ./datasets
        - VGGFACE2
            - train
                - raw
                    - n000810
                    - n000001
                    - ...
                - 4
                - 8
                - ...
                - all_filtered_results.csv
                - all_loose_landmarks_256.csv
            - identity_info.csv
"""

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
                 identity_info_path, filtered_list, use_low_res=False,
                 transform=None):
        """Constructor.

        Args:
            data_dir (str): Directory path containing dataset.
            resolution (int): Specific resolution value to load.
            landmark_info (str): Path of the file having landmark information.
            identity_info (str): Path of the file having identity information.
            filtered_list (str): Path of the file having filtered list
                                 information.
            use_low_res (bool): Use low resolution images or not.
            transform: Augmentation options, Default is None.
                       (e.g. torchvision.transforms.Compose([
                                transform.CenterCrop(10),
                                transform.ToTensor(),
                                ]))
        """
        file_list = []
        dir_list = os.listdir(os.path.join(data_dir, str(resolution)))
        filtered_list = pd.read_csv(filtered_list)
        if use_low_res:
            good_list = filtered_list[filtered_list['category'] !=
                                      'Removed']['filename']
        else:
            good_list = filtered_list[filtered_list['category'] ==
                                      'Good']['filename']

        for ext in ('*.gif', '*.png', '*.jpg'):
            full_path = os.path.normcase(data_dir + f'/{resolution}/*/' + ext)
            file_list.extend(glob.glob(full_path))

        self.file_list = [i for i in file_list if
                          good_list.str.contains(os.path.basename(i)).any()]

        landmark_info = pd.read_csv(landmark_info_path)
        landmark_info = landmark_info[landmark_info['NAME_ID']
                                      .str.contains('|'.join(dir_list))]
        identity_info = pd.read_csv(identity_info_path)
        identity_info = identity_info[identity_info['Class_ID']
                                      .str.contains('|'.join(dir_list))]
        identity_info[' Gender'] = identity_info[' Gender'].apply(
            lambda x: 2 if x == ' f' else 1)

        self.landmark_info = landmark_info
        self.identity_info = identity_info

        self.cls_to_gender = identity_info.set_index('Class_ID')[' Gender']\
                                          .to_dict()
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

        cls_id = name_id.split('/')[0]
        gender = int(self.cls_to_gender[cls_id])

        landmark = self.landmark_info[self.landmark_info['NAME_ID'] ==
                                      name_id].iloc[:, 2:].values.flatten()
        image_arr = np.array(Image.open(image_path))

        sample = {'image': image_arr, 'landmark': landmark, 'gender': gender}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):  # noqa: D105
        return len(self.file_list)
