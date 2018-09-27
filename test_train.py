"""Training test code."""
import sys
import config
from config import Config
from config import EasyDict
import numpy as np
from train import FaceGen
import datetime as dt


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
        self.dataset.num_classes = 4
        self.dataset.num_channels = 3

        # Tranining

        self.train.total_size = 10
        self.train.train_size = 5
        self.train.transition_size = 5

        self.train.net = EasyDict(min_resolution=4,
                                  max_resolution=256,
                                  latent_size=256,
                                  fmap_base=1024,
                                  num_layers=7)

        self.sched.batch_base = 2  # Maximum batch size
        self.sched.batch_dict = {4: 2,
                                 8: 2,
                                 16: 2,
                                 32: 2,
                                 64: 2,
                                 128: 2,
                                 256: 2}  # Resolution-specific overrides


if __name__ == "__main__":
    begin_time = dt.datetime.now()
    env = sys.argv[1] if len(sys.argv) > 2 else 'myconfig'

    if env == 'dev':
        print('With development config,')
        cfg = config.DevelopmentConfig()
    elif env == 'test':
        print('With test config,')
        cfg = config.TestCconfig()
    elif env == 'prod':
        print('With production config,')
        cfg = config.ProductionConfig()
    else:
        print('With my config,')
        cfg = MyConfig()

    print('Running FaceGen()...')
    np.random.seed(cfg.common.random_seed)
    facegen = FaceGen(cfg)
    facegen.train()

    end_time = dt.datetime.now()

    print()
    print("Exiting...", end_time)
    print("Running Time", end_time - begin_time)
