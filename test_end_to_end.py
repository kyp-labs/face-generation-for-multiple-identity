"""Test end to end process."""

import argparse

from torchvision import transforms
from torch.utils.data import DataLoader

from model.model import Generator, Discriminator
from model.unet import UNet
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


def test_end_to_end(args):
    """Test end to end data handling process."""
    batch_size = 1
    res = 256
    lev = 7
    num_domains = 5
    data_shape = [batch_size, 3, res, res]

    transform = transforms.Compose([Normalize(0.5, 0.5),
                                    TargetMask(num_domains),
                                    ScaleNRotate(),
                                    ToTensor()])
    G = Generator(data_shape)
    D = Discriminator(data_shape)
    unet = UNet(3, num_domains)

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

    print(f"lev: {lev}, res: {res}, image: {image.shape}, \
          mask: {mask.shape}, target_id: {target_id}")

    # Generator
    fake_image = G(image, mask, cur_level=lev)
    assert list(fake_image.shape) == [batch_size, 3, res, res], \
        f'Generator: {res, lev} test failed'

    # Discriminator (original)
    cls1 = D(image, lev)
    assert list(cls1.shape) == [batch_size, 1], \
        f'Discriminator: {res, lev} test failed'

    cls2 = D(fake_image, lev)
    assert list(cls2.shape) == [batch_size, 1], \
        f'Discriminator: {res, lev} test failed'

    # Discriminator (pixel-wise)
    mask_pred = unet(image)
    assert list(mask_pred.shape) == [batch_size, num_domains, res, res], \
        f'Pixel-Discriminator: {res, lev} test failed'

    mask_pred_fake = unet(fake_image)
    assert list(mask_pred_fake.shape) == [batch_size, num_domains, res, res], \
        f'Pixel-Discriminator: {res, lev} test failed'


if __name__ == "__main__":
    test_end_to_end(args)
