"""Test model."""


import argparse

from torchvision import transforms
from torch.utils.data import DataLoader

from model.model import Generator, Discriminator
from util.datasets import CelebAHQDataset
from util.custom_transforms import Normalize, CenterSquareMask, \
                                   ScaleNRotate, ToTensor


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',
                    default= '/home/whikwon/Documents/github/download-celebA-HQ/tfrecord/', # noqa E501
                    help='dataset directory')
args = parser.parse_args()


def test_all_level_yes_mask_yes_attr(args):
    """Test model with input image, mask and attributes."""
    transform = transforms.Compose([Normalize(0.5, 0.5),
                                    CenterSquareMask(),
                                    ScaleNRotate(),
                                    ToTensor()])
    batch_size = 1
    num_attrs = 40
    resolutions_to = [4, 8, 8, 16, 16, 32, 32, 64, 64,
                      128, 128, 256, 256]  # 512, 512]
    levels = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5,
              6, 6.5, 7]  # 7.5, 8]
    data_shape = [batch_size, 3, 512, 512]

    G = Generator(data_shape, num_attrs=num_attrs)
    D = Discriminator(data_shape, num_attrs=num_attrs)

    for res, lev in zip(resolutions_to, levels):
        dataset = CelebAHQDataset(args.data_dir, res, transform)
        dataloader = DataLoader(dataset, batch_size, True)
        sample = iter(dataloader).next()  # noqa: B305
        image = sample['image']
        masked_image = sample['masked_image']
        mask = sample['mask']
        attr = sample['attr']
        print(f"level: {lev}, resolution: {res}, image: {masked_image.shape}, \
              mask: {mask.shape}")

        # Generator
        if isinstance(lev, int):
            # training state
            fake_image1 = G(masked_image, attr, mask, lev)
            assert list(fake_image1.shape) == [batch_size, 3, res, res], \
                f'{res, lev} test failed'
        else:
            # transition state
            fake_image2 = G(masked_image, attr, mask, lev)
            assert list(fake_image2.shape) == [batch_size, 3, res, res], \
                f'{res, lev} test failed'

        # Discriminator
        if isinstance(lev, int):
            # training state
            cls1, attr1 = D(image, lev)
            assert list(cls1.shape) == [batch_size, 1], \
                f'{res, lev} test failed'
            assert list(attr1.shape) == [batch_size, num_attrs], \
                f'{res, lev} test failed'
        else:
            # transition state
            cls2, attr2 = D(image, lev)
            assert list(cls2.shape) == [batch_size, 1], \
                f'{res, lev} test failed'
            assert list(attr2.shape) == [batch_size, num_attrs], \
                f'{res, lev} test failed'


def test_all_level_no_mask_yes_attr(args):
    """Test model with input image and attributes."""
    transform = transforms.Compose([Normalize(0.5, 0.5),
                                    CenterSquareMask(),
                                    ScaleNRotate(),
                                    ToTensor()])
    batch_size = 1
    num_attrs = 40
    resolutions_to = [4, 8, 8, 16, 16, 32, 32, 64, 64,
                      128, 128, 256, 256]  # 512, 512]
    levels = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5,
              6, 6.5, 7]  # 7.5, 8]
    data_shape = [batch_size, 3, 512, 512]

    G = Generator(data_shape, use_mask=False, num_attrs=num_attrs)
    D = Discriminator(data_shape, num_attrs=num_attrs)

    for res, lev in zip(resolutions_to, levels):
        dataset = CelebAHQDataset(args.data_dir, res, transform)
        dataloader = DataLoader(dataset, batch_size, True)
        sample = iter(dataloader).next()  # noqa: B305
        image = sample['image']
        masked_image = sample['masked_image']
        mask = sample['mask']
        attr = sample['attr']
        print(f"level: {lev}, resolution: {res}, image: {masked_image.shape}, \
              mask: {mask.shape}")

        # Generator
        if isinstance(lev, int):
            # training state
            fake_image1 = G(masked_image, attr, cur_level=lev)
            assert list(fake_image1.shape) == [batch_size, 3, res, res], \
                f'{res, lev} test failed'
        else:
            # transition state
            fake_image2 = G(masked_image, attr, cur_level=lev)
            assert list(fake_image2.shape) == [batch_size, 3, res, res], \
                f'{res, lev} test failed'

        # Discriminator
        if isinstance(lev, int):
            # training state
            cls1, attr1 = D(image, lev)
            assert list(cls1.shape) == [batch_size, 1], \
                f'{res, lev} test failed'
            assert list(attr1.shape) == [batch_size, num_attrs], \
                f'{res, lev} test failed'
        else:
            # transition state
            cls2, attr2 = D(image, lev)
            assert list(cls2.shape) == [batch_size, 1], \
                f'{res, lev} test failed'
            assert list(attr2.shape) == [batch_size, num_attrs], \
                f'{res, lev} test failed'


def test_all_level_yes_mask_no_attr(args):
    """Test model with input image and mask."""
    transform = transforms.Compose([Normalize(0.5, 0.5),
                                    CenterSquareMask(),
                                    ScaleNRotate(),
                                    ToTensor()])
    batch_size = 1
    resolutions_to = [4, 8, 8, 16, 16, 32, 32, 64, 64,
                      128, 128, 256, 256]  # 512, 512]
    levels = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5,
              6, 6.5, 7]  # 7.5, 8]
    data_shape = [batch_size, 3, 512, 512]

    G = Generator(data_shape, use_attrs=False)
    D = Discriminator(data_shape, use_attrs=False)

    for res, lev in zip(resolutions_to, levels):
        dataset = CelebAHQDataset(args.data_dir, res, transform)
        dataloader = DataLoader(dataset, batch_size, True)
        sample = iter(dataloader).next()  # noqa: B305
        image = sample['image']
        masked_image = sample['masked_image']
        mask = sample['mask']
        print(f"level: {lev}, resolution: {res}, image: {image.shape}, \
              mask: {mask.shape}")

        # Generator
        if isinstance(lev, int):
            # training state
            fake_image1 = G(masked_image, None, mask, cur_level=lev)
            assert list(fake_image1.shape) == [batch_size, 3, res, res], \
                f'{res, lev} test failed'
        else:
            # transition state
            fake_image2 = G(masked_image, None, mask, cur_level=lev)
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


def test_all_level_no_mask_no_attr(args):
    """Test model with input image."""
    transform = transforms.Compose([Normalize(0.5, 0.5),
                                    CenterSquareMask(),
                                    ScaleNRotate(),
                                    ToTensor()])
    batch_size = 1
    resolutions_to = [4, 8, 8, 16, 16, 32, 32, 64, 64,
                      128, 128, 256, 256]  # 512, 512]
    levels = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5,
              6, 6.5, 7]  # 7.5, 8]
    data_shape = [batch_size, 3, 512, 512]

    G = Generator(data_shape, use_attrs=False, use_mask=False)
    D = Discriminator(data_shape, use_attrs=False)

    for res, lev in zip(resolutions_to, levels):
        dataset = CelebAHQDataset(args.data_dir, res, transform)
        dataloader = DataLoader(dataset, batch_size, True)
        sample = iter(dataloader).next()  # noqa: B305
        image = sample['image']
        masked_image = sample['masked_image']
        mask = sample['mask']
        print(f"level: {lev}, resolution: {res}, image: {image.shape}, \
              mask: {mask.shape}")

        # Generator
        if isinstance(lev, int):
            # training state
            fake_image1 = G(masked_image, cur_level=lev)
            assert list(fake_image1.shape) == [batch_size, 3, res, res], \
                f'{res, lev} test failed'
        else:
            # transition state
            fake_image2 = G(masked_image, cur_level=lev)
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
    test_all_level_yes_mask_yes_attr(args)
    test_all_level_yes_mask_no_attr(args)
    test_all_level_no_mask_yes_attr(args)
    test_all_level_no_mask_no_attr(args)
