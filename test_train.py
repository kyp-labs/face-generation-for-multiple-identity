"""Training test code."""
import sys
import config
from config import Config
import numpy as np
from train import FaceGen


class MyConfig(Config):
    """Personal Configuration Class."""

    def __init__(self):
        """Initialize all config variables."""
        super().__init__()
        self.dataset.func = 'util.datasets.VGGFace2Dataset'
        self.dataset.data_dir = './dataset/VGGFACE2/train'
        self.dataset.landmark_path = './dataset/VGGFACE2/bb_landmark/' +\
            'test_loose_landmark.csv'
        self.dataset.identity_path = \
            './dataset/VGGFACE2/test_identity_info.csv'
        self.dataset.num_classes = 3
        self.dataset.num_channels = 3


if __name__ == "__main__":

    env = sys.argv[1] if len(sys.argv) > 2 else 'dev'

    if env == 'dev':
        cfg = config.DevelopmentConfig()
    elif env == 'test':
        cfg = config.TestCconfig()
    elif env == 'prod':
        cfg = config.ProductionConfig()
    else:
        cfg = config.MyConfig()

    print('Running FaceGen()...')
    np.random.seed(cfg.common.random_seed)
    facegen = FaceGen(cfg)
    facegen.train()

    print('Exiting...')
