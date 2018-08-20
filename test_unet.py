"""Test U-Net."""

from torchvision import transforms
from torch.utils.data import DataLoader

from model.unet import UNet
from util.datasets import CelebAHQDataset
from util.custom_transforms import Normalize, CenterSquareMask, \
                                   ScaleNRotate, ToTensor


def test_unet():
    """Test U-Net."""
    res = 256
    transform = transforms.Compose([Normalize(0.5, 0.5),
                                    CenterSquareMask(),
                                    ScaleNRotate(),
                                    ToTensor()])
    batch_size = 5
    num_classes = 5
    num_channels = 3

    dataset = CelebAHQDataset('./test_data/', res, transform)
    dataloader = DataLoader(dataset, batch_size, True)

    sample = iter(dataloader).next()  # noqa: B305
    image = sample['image']

    unet = UNet(num_channels, num_classes)
    out = unet(image)

    assert list(out.shape) == [batch_size, 5, res, res]
