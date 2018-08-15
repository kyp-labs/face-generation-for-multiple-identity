"""util.py.

This file includes enumeration classe and utility functions.
"""
import torch
from enum import Enum
import importlib


# ----------------------------------------------------------------------------
# Enumeration Classes
class Mode(Enum):
    """Mode classes."""

    inpainting = 1
    generation = 2


class TestMode(Enum):
    """Program test mode."""

    full_test = 1
    unit_test = 2


class Gan(Enum):
    """Gan type."""

    gan = 1
    lsgan = 2
    wgan_gp = 3


class Phase(Enum):
    """Type of Phase for training of each layer."""

    transition = 1
    training = 2
    replaying = 3


# VGG16 out_channel : [64, 64, 'M',
#                      128, 128, 'M',
#                      256, 256, 256, 'M',
#                      512, 512, 512, 'M',
#                      512, 512, 512, 'M'],
class Vgg16Layers:
    """Type of layer for Vgg16 Network."""

    conv1_1 = 0
    relu1_1 = 1
    conv1_2 = 2
    relu1_2 = 3
    poo11 = 4

    conv2_1 = 5
    relu2_1 = 6
    conv2_2 = 7
    relu2_2 = 8
    poo12 = 9

    conv3_1 = 10
    relu3_1 = 11
    conv3_2 = 12
    relu3_2 = 13
    conv3_3 = 14
    relu3_3 = 15
    poo13 = 16

    conv4_1 = 17
    relu4_1 = 18
    conv4_2 = 19
    relu4_2 = 20
    conv4_3 = 21
    relu4_3 = 22
    poo14 = 23

    conv5_1 = 24
    relu5_1 = 25
    conv5_2 = 26
    relu5_2 = 27
    conv5_3 = 28
    relu5_3 = 29
    poo15 = 30

    fc6 = 31
    fc7 = 32
    fc8 = 33


class GeneratorLoss:
    """Loss of generator.

    Attributes:
        g_loss : loss of generator
        g_adver_loss : adversarial loss
        recon_loss : reconstruction loss
        feat_loss : feature loss
        bdy_loss : boundary loss

    """

    def __init__(self):
        """Init attributes."""
        self.g_loss = 0
        self.g_adver_loss = 0
        self.recon_loss = 0
        self.feat_loss = 0
        self.bdy_loss = 0


class DiscriminatorLoss:
    """Loss of discriminator.

    Attributes:
        d_loss : loss of discriminator
        d_adver_loss : adversarial loss
        d_adver_loss_syn : adversarial loss of synthesized image (fake)
        d_adver_loss_real : adversarial loss of real image (real)
        att_loss : attribute loss
        gradient_penalty_loss : gradient penalty of WGAN GP

    """

    def __init__(self):
        """Init attributes."""
        self.d_loss = 0
        self.d_adver_loss = 0
        self.d_adver_loss_syn = 0
        self.d_adver_loss_real = 0
        self.att_loss = 0
        self.gradient_penalty_loss = 0


class GeneratorLossHistory:
    """Loss history for generator.

    Attributes:
        g_loss_hist : history for generator loss
        g_adver_loss_hist : history for adversarial loss
        recon_loss_hist : history for reconstruction loss
        feat_loss_hist : history for feature loss
        bdy_loss_hist : history for boundary loss

    """

    def __init__(self):
        """Init attributes."""
        self.g_loss_hist = []
        self.g_adver_loss_hist = []
        self.recon_loss_hist = []
        self.feat_loss_hist = []
        self.bdy_loss_hist = []

    def append(self, g_losses):
        """Append new loss to history.

        Args:
            g_losses: new loss of generator
        """
        self.g_loss_hist.append(g_losses.g_loss.detach())
        self.g_adver_loss_hist.append(g_losses.g_adver_loss)
        self.recon_loss_hist.append(g_losses.recon_loss)
        self.feat_loss_hist.append(g_losses.feat_loss)
        self.bdy_loss_hist.append(g_losses.bdy_loss)

    def len(self):
        """Length of history."""
        return self.d_loss_hist.len()


class DiscriminatorLossHistory:
    """Loss history for discriminator.

    Attributes:
        d_loss_hist : history for generator loss
        d_adver_loss_hist : history for adversarial loss
        d_adver_loss_syn_hist : history for adversarial loss of synthesized img
        d_adver_loss_real_hist : history for adversarial loss of real images
        att_loss_hist : history for attribute loss
        gradient_penalty_hist : history for gradient penalty

    """

    def __init__(self):
        """Init attributes."""
        self.d_loss_hist = []
        self.d_adver_loss_hist = []
        self.d_adver_loss_syn_hist = []
        self.d_adver_loss_real_hist = []
        self.att_loss_hist = []
        self.gradient_penalty_hist = []

    def append(self, d_losses):
        """Append new loss to history.

        Args:
            d_losses: new loss of discriminator

        """
        self.d_loss_hist.append(d_losses.d_loss.detach())
        self.d_adver_loss_hist.append(d_losses.d_adver_loss)
        self.d_adver_loss_syn_hist.append(d_losses.d_adver_loss_syn)
        self.d_adver_loss_real_hist.append(d_losses.d_adver_loss_real)
        self.att_loss_hist.append(d_losses.att_loss)
        self.gradient_penalty_hist.append(d_losses.gradient_penalty)

    def len(self):
        """Length of history."""
        return self.d_loss_hist.len()


# ----------------------------------------------------------------------------
# Utilities for Tensor and Other Types
def tofloat(use_cuda, var):
    """Type conversion to cuda accoding to cuda use.

    Args:
        use_cuda: flag for cuda use
        var: type converted variable

    """
    if use_cuda:
        var = var.cuda().float()
    var = var.float()  # should be deleted
    return var


def numpy2tensor(use_cuda, var):
    """Type conversion of numpy to tensor accoding to cuda use.

    Args:
        use_cuda: flag for cuda use
        var: type converted variable

    """
    var = torch.Tensor(torch.from_numpy(var))
    if use_cuda:
        var = var.cuda()
    return var


def tensor2numpy(use_cuda, var):
    """Type conversion of tensor to numpy accoding to cuda use.

    Args:
        use_cuda: flag for cuda use
        var: type converted variable

    """
    if use_cuda:
        return var.cpu().data.numpy()
    return var.data.numpy()


def get_data(d):
    """Get data in native type from tensor."""
    return d.data if isinstance(d, torch.Tensor) else d


# ----------------------------------------------------------------------------
# Utilities for Image Normlalization
def rescale(img, from_range, to_range):
    """Rescale value of image in new reange.

    Args:
        img: input image
        from_range: value range before change [0.0, 255]
        to_range: value range after change [-1.0, 1.0]

    """
    from_diff = from_range[1] - from_range[0]
    to_diff = to_range[1] - to_range[0]
    normalized_img = (img - from_range[0])/from_diff
    scaled_img = normalized_img*to_diff + to_range[0]
    return scaled_img


def normalize(img, from_range, to_range):
    """Normalize value of image in new reange.

    Args:
        img: input image
        from_range: value range before change [0.0, 255.0]
        to_range: value range after change [-1.0, 1.0]

    """
    return rescale(img, from_range, to_range)


def denormalize(img, from_range, to_range):
    """Denormalize value of image in new reange.

    Args:
        img: input image
        from_range: value range before change [-1.0, 1.0]
        to_range: value range after change [0.0, 255.0]

    """
    return rescale(img, from_range, to_range)


def normalize_min_max(img):
    """Normalize value of image in [min, max] range."""
    img = img - torch.min(img)
    img = img/torch.max(img) * 2.0 - 1.0

    return img


# ----------------------------------------------------------------------------
# Utilities for importing modules and objects by name.
def import_module(module_or_obj_name):
    """Import module.

    Args:
        module_or_obj_name: module or object name to import

    """
    parts = module_or_obj_name.split('.')
    parts[0] = {'np': 'numpy', 'tf': 'tensorflow'}.get(parts[0], parts[0])
    for i in range(len(parts), 0, -1):
        try:
            module = importlib.import_module('.'.join(parts[:i]))
            relative_obj_name = '.'.join(parts[i:])
            return module, relative_obj_name
        except ImportError:
            pass
    raise ImportError(module_or_obj_name)


def find_obj_in_module(module, relative_obj_name):
    """Find obj in module.

    Args:
        module: module name to be found
        relative_obj_name : module name to search

    """
    obj = module
    for part in relative_obj_name.split('.'):
        obj = getattr(obj, part)
    return obj


def import_obj(obj_name):
    """Import object.

    Args:
        obj_name : object name to search

    """
    module, relative_obj_name = import_module(obj_name)
    return find_obj_in_module(module, relative_obj_name)


def call_func_by_name(*args, func=None, **kwargs):
    """Call function by namee.

    Args:
        args : function arguments
        func : function name
        kwargs : function arguments in dictionary type

    """
    assert func is not None
    return import_obj(func)(*args, **kwargs)
