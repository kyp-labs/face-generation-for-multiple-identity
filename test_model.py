"""Test model."""

import argparse

from torchvision import transforms
from torch.utils.data import DataLoader

from model.model import Generator, Discriminator
from util.datasets import VGGFace2Dataset
from util.custom_transforms import Normalize, TargetMask, \
                                   ScaleNRotate, ToTensor


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',
                    default= './dataset/VGGFACE2/train', # noqa E501
                    help='dataset directory')
parser.add_argument('--landmark_info_path',
                    default= './dataset/VGGFACE2/bb_landmark/test_loose_landmark.csv', # noqa E501
                    help='dataset directory')
parser.add_argument('--identity_info_path',
                    default= './dataset/VGGFACE2/test_identity_info.csv', # noqa E501
                    help='dataset directory')
args = parser.parse_args()


def test_all_level_yes_mask(args):
    """Test model with input image and mask."""
    transform = transforms.Compose([Normalize(0.5, 0.5),
                                    TargetMask(5),
                                    ScaleNRotate(),
                                    ToTensor()])
    batch_size = 1
    resolutions_to = [4, 8, 8, 16, 16, 32, 32, 64, 64,
                      128, 128, 256, 256]  # 512, 512]
    levels = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5,
              6, 6.5, 7]  # 7.5, 8]
    data_shape = [batch_size, 3, 512, 512]

    G = Generator(data_shape)
    D = Discriminator(data_shape)

    for res, lev in zip(resolutions_to, levels):
        dataset = VGGFace2Dataset(args.data_dir,
                                  res,
                                  args.landmark_info_path,
                                  args.identity_info_path,
                                  transform)
        dataloader = DataLoader(dataset, batch_size, True)
        sample = iter(dataloader).next()  # noqa: B305
        image = sample['image']
        mask = sample['mask']
        target_id = sample['target_id']
        print(f"level: {lev}, resolution: {res}, image: {image.shape}, \
              mask: {mask.shape}, target_id: {target_id}")

        # Generator
        if isinstance(lev, int):
            # training state
            fake_image1 = G(image, mask, cur_level=lev)
            assert list(fake_image1.shape) == [batch_size, 3, res, res], \
                f'{res, lev} test failed'
        else:
            # transition state
            fake_image2 = G(image, mask, cur_level=lev)
            assert list(fake_image2.shape) == [batch_size, 3, res, res], \
                f'{res, lev} test failed'

        # Discriminator
        if isinstance(lev, int):
            # training state
            cls1 = D(image, lev)
            assert list(cls1.shape) == [batch_size, 1], \
                f'{res, lev} test failed'
        else:
            # transition state
            cls2 = D(image, lev)
            assert list(cls2.shape) == [batch_size, 1], \
                f'{res, lev} test failed'


def test_all_level_no_mask(args):
    """Test model with input image."""
    transform = transforms.Compose([Normalize(0.5, 0.5),
                                    TargetMask(5),
                                    ScaleNRotate(),
                                    ToTensor()])
    batch_size = 1
    resolutions_to = [4, 8, 8, 16, 16, 32, 32, 64, 64,
                      128, 128, 256, 256]  # 512, 512]
    levels = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5,
              6, 6.5, 7]  # 7.5, 8]
    data_shape = [batch_size, 3, 512, 512]

    G = Generator(data_shape, use_mask=False)
    D = Discriminator(data_shape)

    for res, lev in zip(resolutions_to, levels):
        dataset = VGGFace2Dataset(args.data_dir,
                                  res,
                                  args.landmark_info_path,
                                  args.identity_info_path,
                                  transform)
        dataloader = DataLoader(dataset, batch_size, True)
        sample = iter(dataloader).next()  # noqa: B305
        image = sample['image']
        mask = sample['mask']
        target_id = sample['target_id']
        print(f"level: {lev}, resolution: {res}, image: {image.shape}, \
              mask: {mask.shape}, target_id: {target_id}")

        # Generator
        if isinstance(lev, int):
            # training state
            fake_image1 = G(image, cur_level=lev)
            assert list(fake_image1.shape) == [batch_size, 3, res, res], \
                f'{res, lev} test failed'
        else:
            # transition state
            fake_image2 = G(image, cur_level=lev)
            assert list(fake_image2.shape) == [batch_size, 3, res, res], \
                f'{res, lev} test failed'

        # Discriminator
        if isinstance(lev, int):
            # training state
            cls1 = D(image, lev)
            assert list(cls1.shape) == [batch_size, 1], \
                f'{res, lev} test failed'
        else:
            # transition state
            cls2 = D(image, lev)
            assert list(cls2.shape) == [batch_size, 1], \
                f'{res, lev} test failed'


if __name__ == "__main__":
    test_all_level_yes_mask(args)
    test_all_level_no_mask(args)
