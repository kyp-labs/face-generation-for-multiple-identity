"""config.py.

This module includes ReplayMemory classe
which matains old input data to replay discriminator with them.

"""
from util.util import Gan
from util.util import Mode
from util.util import TestMode


# ----------------------------------------------------------------------------
# Convenience class that behaves exactly like dict(), but allows accessing
# the keys and values using the attribute syntax, i.e., "mydict.key = value".

class EasyDict(dict):
    """Custom dictionary class for configuration."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        """Get attribute."""
        return self[name]

    def __setattr__(self, name, value):
        """Set attribute."""
        self[name] = value

    def __delattr__(self, name):
        """Delete attribute."""
        del self[name]


class Config():
    """Configuration Class."""

    def __init__(self):
        """Initialize all configuration variables."""
        # Common Parameters
        self.common = EasyDict()
        self.common.random_seed = 1000  # Global random seed.
        self.common.test_mode = TestMode.full_test  # {unit_test, full_test}

        # Environment
        self.env = EasyDict()
        self.env.num_gpus = 1

        # DataSet
        self.dataset = EasyDict()
        self.dataset.func = 'util.datasets.VGGFace2Dataset'
        self.dataset.data_dir = './dataset/VGGFACE2/train'
        self.dataset.landmark_path = './dataset/VGGFACE2/bb_landmark/' +\
            'test_loose_landmark.csv'
        self.dataset.identity_path = \
            './dataset/VGGFACE2/test_identity_info.csv'
        self.dataset.num_classes = 3
        self.dataset.num_channels = 3

        # Tranining
        self.test1_train = EasyDict(D_repeats=1,
                                    total_size=100,
                                    train_size=50,
                                    transition_size=50,
                                    dataset_unit=1)
        self.test2_train = EasyDict(D_repeats=4,
                                    total_size=1000,
                                    train_size=500,
                                    transition_size=500,
                                    dataset_unit=1)
        self.test3_train = EasyDict(D_repeats=1,
                                    total_size=50000,
                                    train_size=25000,
                                    transition_size=25000,
                                    dataset_unit=1)

        self.train = self.test1_train
        self.train.net = EasyDict(min_resolution=4,
                                  max_resolution=256,
                                  latent_size=256,
                                  fmap_base=1024,
                                  num_layers=7)

        self.train.use_mask = True  # {inpainting , generation} mode
        self.train.mode = Mode.generation  # {inpainting , generation} mode
        if self.common.test_mode == TestMode.unit_test:
            self.train.forced_stop = True
        else:
            self.train.forced_stop = False
        self.train.forced_stop_resolution = 4  # {inpainting , generation} mode

        # Training Scheduler
        self.sched = EasyDict()
        self.sched.batch_base = 32  # Maximum batch size
        self.sched.batch_dict = {4: 2,
                                 8: 2,
                                 16: 2,
                                 32: 2,
                                 64: 2,
                                 128: 2,
                                 256: 2}  # Resolution-specific overrides
        self.sched.batch_dict2 = {4: 64,
                                  8: 32,
                                  16: 16,
                                  32: 16,
                                  64: 4,
                                  128: 4,
                                  256: 2}  # Resolution-specific overrides

        # Replay
        self.replay = EasyDict()
        self.replay.enabled = False
        self.replay.replay_count = 100
        self.replay.max_memory_size = 256
        self.replay.max_memory_size_dict = {4: 256,
                                            8: 256,
                                            16: 256,
                                            32: 256,
                                            64: 128,
                                            128: 128,
                                            256: 64}  # 8 times batch size

        # Loss
        self.loss = EasyDict()
        self.loss.use_feat_loss = False

        # type of gan {ga, lsgan, wgan gp, sngan}
        self.loss.gan = Gan.sngan
        # weight of syn images' loss of D
        self.loss.alpha_adver_loss_syn = 1.0
        # weight for target area of reconstruction loss (0.7)
        self.loss.alpha_recon = 0.7
        # weight of gradient panelty (ref source = 10)
        self.loss.lambda_GP = 10.0

        # weight of reconstruction loss (paper = 500)
        self.loss.lambda_recon = 500.0
        # weight of feature loss (paper = 10)
        self.loss.lambda_feat = 10.0
        # weight of boundary loss(paper = 5000)
        self.loss.lambda_bdy = 5000.0
        # weight of cycle consistency loss
        self.loss.lambda_cycle = 500.0
        self.loss.lambda_pixel = 1
        # mean filter size for calculation of boudnary loss
        self.loss.mean_filter_size = 7

        # Optimizer
        self.optimizer = EasyDict()
        self.optimizer.G_opt = EasyDict(beta1=0.5,  # generator optimizer
                                        beta2=0.99,
                                        epsilon=1e-8)
        self.optimizer.D_opt = EasyDict(beta1=0.5,  # discriminator optimizer
                                        beta2=0.99,
                                        epsilon=1e-8)

        # Learning Rate
        self.optimizer.lrate = EasyDict()
        self.optimizer.lrate.rampup_rate = 0.2
        self.optimizer.lrate.rampdown_rate = 0.2
        self.optimizer.lrate.G_base = 0.0002  # 1e-3
        self.optimizer.lrate.D_base = 0.0002  # 1e-3
        self.optimizer.lrate.G_dict = {1024: 0.0015}
        self.optimizer.lrate.D_dict = EasyDict(self.optimizer.lrate.G_dict)

        # Snapshot
        self.snapshot = EasyDict()
        self.snapshot.exp_dir = './exp'  # experiment dir
        self.snapshot.sample_freq = 128  # sample frequency, 500
        self.snapshot.sample_freq_dict = {4: 128,
                                          8: 256,
                                          16: 512,
                                          32: 512,
                                          64: 1024,
                                          128: 1024,
                                          256: 1024}
        self.snapshot.rows_map = {64: 8,
                                  32: 8,
                                  16: 4,
                                  8: 2,
                                  4: 2,
                                  2: 1,
                                  1: 1}  # rows per batch size
        self.snapshot.enable_threading = False

        # Model Save & Restore
        self.checkpoint = EasyDict()
        self.checkpoint.restore = True
        self.checkpoint.restore_dir = ''  # restore from which exp dir
        self.checkpoint.which_file = ''  # restore from which file

        self.checkpoint.save_freq = 128  # save model frequency
        self.checkpoint.save_freq_dict = self.snapshot.sample_freq_dict

        # Loggingalpha_adver
        self.logging = EasyDict()
        self.logging.log_dir = './logs'


class DevelopmentConfig(Config):
    """Development Configuration Class."""

    def __init__(self):
        """Initialize all configuration variables."""
        super().__init__()
        self.dataset.func = 'util.datasets.VGGFace2Dataset'
        self.dataset.data_dir = './dataset/VGGFACE2/train'
        self.dataset.landmark_path = './dataset/VGGFACE2/train/' +\
            'all_loose_landmarks_256.csv'
        self.dataset.identity_path =\
            './dataset/VGGFACE2/identity_info.csv'
        self.dataset.filtering_path =\
            './dataset/VGGFACE2/train/all_filtered_results.csv'
        self.dataset.num_classes = 3
        self.dataset.num_channels = 3


class TestConfig(Config):
    """Test Configuration Class."""

    def __init__(self):
        """Initialize all configuration variables."""
        super().__init__()
        self.dataset.func = 'util.datasets.VGGFace2Dataset'
        self.dataset.data_dir = './dataset/VGGFACE2/train'
        self.dataset.landmark_path = './dataset/VGGFACE2/train/' +\
            'all_loose_landmarks_256.csv'
        self.dataset.identity_path =\
            './dataset/VGGFACE2/identity_info.csv'
        self.dataset.filtering_path =\
            './dataset/VGGFACE2/train/all_filtered_results.csv'
        self.dataset.num_classes = 3
        self.dataset.num_channels = 3


class ProductionConfig(Config):
    """Production Configuration Class."""

    def __init__(self):
        """Initialize all configuration variables."""
        super().__init__()
        self.dataset.func = 'util.datasets.VGGFace2Dataset'
        self.dataset.data_dir = './dataset/VGGFACE2/train'
        self.dataset.landmark_path = './dataset/VGGFACE2/train/' +\
            'all_loose_landmarks_256.csv'
        self.dataset.identity_path =\
            './dataset/VGGFACE2/identity_info.csv'
        self.dataset.filtering_path =\
            './dataset/VGGFACE2/train/all_filtered_results.csv'
        self.dataset.num_classes = 3
        self.dataset.num_channels = 3


'''
config = EasyDict()
config.common = common
config.env = env
config.dataset = dataset
config.train = train
config.sched = sched
config.replay = replay
config.loss = loss
config.optimizer = optimizer
config.snapshot = snapshot
config.checkpoint = checkpoint
config.logging = logging
'''
