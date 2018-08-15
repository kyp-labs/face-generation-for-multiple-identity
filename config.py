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


# Common Parameters
common = EasyDict()
common.desc = 'facegen'  # Description string included in result subdir name.
common.random_seed = 1000  # Global random seed.
common.test_mode = TestMode.full_test  # {unit_test, full_test}


# Environment
env = EasyDict()
env.num_gpus = 1

# DataSet
dataset = EasyDict(func='util.datasets.CelebADataset',
                   data_dir='test_data/n000106/',
                   num_channels=3,
                   attibute_size=4)

# Tranining
test1_train = EasyDict(D_repeats=4,
                       total_size=16,
                       train_size=8,
                       transition_size=8,
                       dataset_unit=1)
test2_train = EasyDict(D_repeats=4,
                       total_size=1000,
                       train_size=500,
                       transition_size=500,
                       dataset_unit=1)
test3_train = EasyDict(D_repeats=1,
                       total_size=50000,
                       train_size=25000,
                       transition_size=25000,
                       dataset_unit=1)
train = test3_train
train.net = EasyDict(min_resolution=4,
                     max_resolution=512,
                     latent_size=512,
                     fmap_base=2048,
                     fmap_decay=1.0)

train.use_mask = True  # {inpainting , generation} mode
train.use_attr = True  # {inpainting , generation} mode
train.mode = Mode.inpainting  # {inpainting , generation} mode
if common.test_mode == TestMode.unit_test:
    train.forced_stop = True
else:
    train.forced_stop = False
train.forced_stop_resolution = 4  # {inpainting , generation} mode

# Training Scheduler
sched = EasyDict()
sched.batch_base = 32  # Maximum batch size
sched.batch_dict = {4: 64,
                    8: 32,
                    16: 16,
                    32: 16,
                    64: 4,
                    128: 4,
                    256: 2,
                    512: 1,
                    1024: 1}  # Resolution-specific overrides

# Replay
replay = EasyDict()
replay.enabled = False
replay.replay_count = 100
replay.max_memory_size = 256
replay.max_memory_size_dict = {4: 256,
                               8: 256,
                               16: 256,
                               32: 256,
                               64: 128,
                               128: 128,
                               256: 64,
                               512: 32,
                               1024: 16}  # 8 times batch size

# Loss
loss = EasyDict()
loss.use_feat_loss = False

loss.gan = Gan.wgan_gp  # type of gan {wgan gp, lsgan, gan}
loss.alpha_adver_loss_syn = 1.0  # weight of syn images' loss of D
loss.alpha_recon = 0.7  # weight for mask area of reconstruction loss (0.7)
loss.lambda_GP = 10.0  # weight of gradient panelty (ref source = 10)

loss.lambda_recon = 500.0  # weight of reconstruction loss (paper = 500)
loss.lambda_feat = 10.0  # weight of feature loss (paper = 10)
loss.lambda_bdy = 5000.0  # weight of boundary loss(paper = 5000)
loss.lambda_attr = 2.0  # weight of attribute loss (paper = 2)
loss.mean_filter_size = 7  # mea filter size for calculation of boudnary loss

# Optimizer
optimizer = EasyDict()
optimizer.G_opt = EasyDict(beta1=0.5,
                           beta2=0.99,
                           epsilon=1e-8)  # generator optimizer
optimizer.D_opt = EasyDict(beta1=0.5,
                           beta2=0.99,
                           epsilon=1e-8)  # discriminator optimizer

# Learning Rate
optimizer.lrate = EasyDict()
optimizer.lrate.rampup_rate = 0.2
optimizer.lrate.rampdown_rate = 0.2
optimizer.lrate.G_base = 0.0002  # 1e-3
optimizer.lrate.D_base = 0.0002  # 1e-3
optimizer.lrate.G_dict = {1024: 0.0015}
optimizer.lrate.D_dict = EasyDict(optimizer.lrate.G_dict)


# Snapshot
snapshot = EasyDict()
snapshot.exp_dir = './exp'  # experiment dir
snapshot.sample_freq = 128  # sample frequency, 500
snapshot.sample_freq_dict = {4: 128,
                             8: 256,
                             16: 512,
                             32: 512,
                             64: 1024,
                             128: 256,
                             512: 2048,
                             1024: 2048}
snapshot.rows_map = {64: 8,
                     32: 8,
                     16: 4,
                     8: 2,
                     4: 2,
                     2: 1}  # rows per batch size
snapshot.enable_threading = True
snapshot.draw_plot = False


# Model Save & Restore
checkpoint = EasyDict()
checkpoint.restore = True
checkpoint.restore_dir = ''  # restore from which exp dir
checkpoint.which_file = ''  # restore from which file

checkpoint.save_freq = 128  # save model frequency
checkpoint.save_freq_dict = snapshot.sample_freq_dict

# Loggingalpha_adver
logging = EasyDict()
logging.log_dir = './logs'


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
