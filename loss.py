"""loss.py.

This module includes following loss related classes.

1. FaceGenLoss class :
   calculates 5 types losses of generator and discriminator.

   - adversarial loss : gan, wgan gp
   - reconstruction loss
   - boundary loss
   - feature loss
   - pixelwise classcification loss

2. Vgg16FeatureExtractor class :
    has a VGG16 model pretrained ImageNet
    and a method of extraction of any feature map.

3. MeaFilter class :
    supports mean filtering of any image.

"""

import torch
import torch.nn.functional as F
from torchvision.models import vgg16
from torch import autograd
from torch import nn

import util.util as util
from util.util import Gan
from util.util import GeneratorLoss
from util.util import DiscriminatorLoss
from util.util import Vgg16Layers


class FaceGenLoss():
    """FaceGenLoss classes.

    Attributes:
        use_cuda : flag for cuda use
        gpu (int32) : # of gpus
        alpha_adver_loss_syn (int32) : weight of syn images' loss of D
        alpha_recon (int32) : weight for mask area of reconstruction loss
        lambda_GP (int32) : weight of gradient panelty
        lambda_recon (int32) :weight of reconstruction loss
        lambda_feat (int32) : weight of feature loss
        lambda_bdy (int32) : weight of boundary loss
        g_losses : losses of generator
        d_losses : losses of discriminator
        p_losses : losses of pixelwise classifier
        gan (enum) : type of gan {wgan gp, lsgan, gan}
        vgg16 : VGG16 feature extractor
        adver_loss_func : adversarial loss function

    """

    def __init__(self, config, use_cuda=False, gpu=-1):
        """Class initializer.

        Steps:
            1. Read loss params from self.config.py
            2. Create loss functions
            3. Create VGG16 model and feature extractor

        """
        self.config = config
        self.pytorch_loss_use = True

        self.use_cuda = use_cuda
        self.gpu = gpu

        self.alpha_adver_loss_syn = self.config.loss.alpha_adver_loss_syn
        self.alpha_recon = self.config.loss.alpha_recon

        self.lambda_GP = self.config.loss.lambda_GP
        self.lambda_recon = self.config.loss.lambda_recon
        self.lambda_feat = self.config.loss.lambda_feat
        self.lambda_bdy = self.config.loss.lambda_bdy
        self.lambda_cycle = self.config.loss.lambda_cycle
        self.lambda_pixel = self.config.loss.lambda_pixel

        self.g_losses = GeneratorLoss()
        self.d_losses = DiscriminatorLoss()

        self.gan = self.config.loss.gan
        self.create_loss_functions(self.gan)

        # for computing feature loss
        if self.config.loss.use_feat_loss:
            # Vgg16 ImageNet Pretrained Model
            self.vgg16 = Vgg16FeatureExtractor()

        self.register_on_gpu()

    def register_on_gpu(self):
        """Set vgg16 to cuda according to gpu availability."""
        if self.use_cuda:
            if self.config.loss.use_feat_loss:
                self.vgg16.cuda()

    def create_loss_functions(self, gan):
        """Create loss functions.

        1. create adversarial loss function
        2. create attribute loss function

        Args:
            gan: type of gan {wgan gp, lsgan, gan}

        """
        # adversarial loss function
        if gan == Gan.sngan:
            self.adver_loss_func = lambda p, t: (-2.0*t+1.0) * torch.mean(p)
        elif gan == Gan.wgan_gp:
            self.adver_loss_func = lambda p, t: (-2.0*t+1.0) * torch.mean(p)
        elif gan == Gan.lsgan:
            self.adver_loss_func = lambda p, t: torch.mean((p-t)**2)
        elif gan == Gan.gan:  # 1e-8 torch.nn.BCELoss()
            self.adver_loss_func = torch.nn.BCELoss()
        else:
            raise ValueError('Invalid/Unsupported GAN: %s.' % gan)

    def calc_adver_loss(self, prediction, target):
        """Calculate adversarial loss.

        Args:
            prediction: prediction of discriminator
            target: target label {True, False}
            w: weight of adversarial loss

        """
        if self.gan == Gan.gan and self.pytorch_loss_use:
            N = prediction.shape[0]
            if target is True:
                target = util.tofloat(self.use_cuda, torch.ones(N))
            else:
                target = util.tofloat(self.use_cuda, torch.zeros(N))
            return self.adver_loss_func(prediction, target)
        else:
            return self.adver_loss_func(prediction, target)

    def calc_gradient_penalty(self, D, cur_level, real, syn):
        """Calc gradient penalty of wgan gp.

        Args:
            D: discriminator
            cur_level: progress indicator of progressive growing network
            real: real images
            syn: synthesized images

        """
        N, C, H, W = real.shape

        alpha = torch.rand(N, 1)
        alpha = alpha.expand(N,
                             real.nelement()/N).contiguous().view(N, C, H, W)
        alpha = util.tofloat(self.use_cuda, alpha)

        syn = syn.detach()
        interpolates = alpha * real + (1.0 - alpha) * syn

        interpolates = util.tofloat(self.use_cuda, interpolates)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        cls_interpol, pixel_cls_interpol = D(interpolates, cur_level)

        cls_interpol = cls_interpol[:1, :]  # temporary code
        grad_outputs = util.tofloat(self.use_cuda,
                                    torch.ones(cls_interpol.size()))
        gradients = autograd.grad(outputs=cls_interpol,
                                  inputs=interpolates,
                                  grad_outputs=grad_outputs,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]

        gradients = gradients.view(gradients.size(0), -1)
        return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean() * self.lambda_GP

    def calc_feat_loss(self, real, syn):
        """Calculate feature loss.

        Args:
            real : real images
            syn : synthesized images

        """
        if self.config.loss.use_feat_loss is False:
            return 0

        # get activation of relu2_2
        N, C, H, W = real.shape
        # if H < 16 :
        #    return 0

        real_fmap = self.vgg16(real.detach(), Vgg16Layers.relu2_2)
        syn_fmap = self.vgg16(syn.detach(), Vgg16Layers.relu2_2)

        feat_loss = real_fmap - syn_fmap
        feat_loss = ((feat_loss.norm(2, dim=1) - 1.0) ** 2).mean()
        return feat_loss

    def calc_recon_loss(self, real, real_mask, syn, obs_mask):
        """Calculate reconstruction loss.

        Args:
            real (tensor) : real images
            real_mask (tensor) : domain masks of real images
            syn (tensor) : synthesized images
            obs_mask (tensor) : domain masks of observed images

        """
        N, C, H, W = real.shape

<<<<<<< HEAD
        # target area of domain mask
        mask = 1 - util.tofloat(self.use_cuda, real_mask == obs_mask)
=======
        # domain area of input image
        mask = util.tofloat(self.use_cuda, real_mask == obs_mask)
>>>>>>> a9efead86d4d8f0a05ff5f23ddd79ab338d1d29d
        # mask *= obs_mask
        mask = mask.repeat((1, C, 1, 1))

        # L1 norm
        alpha = self.alpha_recon
        recon_loss = (alpha * mask * (real - syn)).norm(1) + \
                     ((1 - alpha) * (1 - mask) * (real - syn)).norm(1)

        recon_loss = recon_loss/N

        return recon_loss

    def calc_bdy_loss(self, real, real_mask, syn, obs_mask):
        """Calculate boundary loss.

        Args:
            real (tensor) : real images
            real_mask (tensor) : domain masks of real images
            syn (tensor) : synthesized images
            obs_mask (tensor) : domain masks of observed images

        """
        # blurring mask boundary
        N, C, H, W = obs_mask.shape

<<<<<<< HEAD
        # target area of domain mask
        mask = 1 - util.tofloat(self.use_cuda, real_mask == obs_mask)
=======
        # domain area of input image
        mask = util.tofloat(self.use_cuda, real_mask == obs_mask)
>>>>>>> a9efead86d4d8f0a05ff5f23ddd79ab338d1d29d
        # mask *= obs_mask
        mask = mask.repeat((1, C, 1, 1))

        # if H < 16:
        #    return 0

        mean_filter = MeanFilter(mask.shape, self.config.loss.mean_filter_size)
        if self.use_cuda:
            mean_filter.cuda()

        w = mean_filter(mask)
        w = w * mask  # weights of mask range are 0
        w_ext = w.repeat((1, C, 1, 1))

        w_ext = util.tofloat(self.use_cuda, w_ext)

        bdy_loss = (w_ext * (real - syn)).norm(1)
        bdy_loss = bdy_loss.sum()/N

        return bdy_loss

    def calc_cycle_loss(self, G, cur_level, real, real_mask, syn):
        """Calculate cycle consistency loss.

        Args:
            G: generator
            cur_level: progress indicator of progressive growing network
            real (tensor) : real images
            real_mask (tensor) : domain masks of real images
            obs_mask (tensor) : domain masks of observed images

        """
        N, C, H, W = real.shape

        pred_real = G(syn,
                      mask=real_mask,
                      cur_level=cur_level)

        # L1 norm
        cycle_loss = F.l1_loss(pred_real, real, size_average=True)
        return cycle_loss

    def calc_G_loss(self,
                    G,
                    cur_level,
                    real,
                    real_mask,
                    obs,
                    obs_mask,
                    syn,
                    cls_real,
                    cls_syn,
                    pixel_cls_real,
                    pixel_cls_syn):
        """Calculate Generator loss.

        Args:
            G : generator
            cur_level (float32) : progress indicator of
                                  progressive growing network
            real (tensor) : real images
            real_mask (tensor) : domain masks of real images
            obs (tensor) : observed images
            obs_mask (tensor) : domain masks of observed images
            syn (tensor) : synthesized images
            cls_real (tensor) : classes for real images
            cls_syn (tensor) : classes for synthesized images
            pixel_cls_real (tensor) : pixelwise classes for real images
            pixel_cls_syn (tensor) : pixelwise classes for synthesized images

        """
        # adversarial loss
        self.g_losses.g_adver_loss = self.calc_adver_loss(cls_syn, True)
        # reconstruction loss
        self.g_losses.recon_loss = self.calc_recon_loss(real,
                                                        real_mask,
                                                        syn,
                                                        obs_mask)
        # feature loss
        self.g_losses.feat_loss = self.calc_feat_loss(real, syn)
        # boundary loss
        self.g_losses.bdy_loss = self.calc_bdy_loss(real,
                                                    real_mask,
                                                    syn,
                                                    obs_mask)
        # cycle consistency loss
        self.g_losses.cycle_loss = self.calc_cycle_loss(G,
                                                        cur_level,
                                                        real,
                                                        real_mask,
                                                        syn)

        # pixelwise classification koss
        self.g_losses.pixel_loss = 0
        # if pixel_cls_syn is None:
        #    self.g_losses.pixel_loss = 0
        # else:
        #    self.g_losses.pixel_loss = \
        #        self.cross_entropy2d(pixel_cls_syn, obs_mask)

        self.g_losses.g_loss = self.g_losses.g_adver_loss + \
            self.lambda_recon*self.g_losses.recon_loss + \
            self.lambda_feat*self.g_losses.feat_loss + \
            self.lambda_bdy*self.g_losses.bdy_loss + \
            self.lambda_cycle*self.g_losses.cycle_loss + \
            self.lambda_pixel*self.g_losses.pixel_loss

        return self.g_losses

    def cross_entropy2d(self,
                        predict,
                        target,
                        weight=None,
                        size_average=False):
        """Calculate cross entropy for segmentation class.

        Args:
            predict (tensor) : [batch_size, num_channels, height, width]
                                prediction of pixelwise classifier
            target (tensor) : [batch_size, num_channels, height, width]
                                target label {class id}
            weight (tensor) : [# of classes]
                    weight of pixels

        Return:
            loss (scalar) : cross entropy loss

        """
        assert not (predict is None or target is None)

        log_p = F.log_softmax(predict, dim=1)

        N, C, H, W = target.shape
        target = target.permute(0, 2, 3, 1).view(N, H, W)
        target = target.type(torch.cuda.LongTensor)

        loss = F.nll_loss(log_p,
                          target,
                          weight=weight,
                          size_average=size_average)
        return loss

    def calc_D_loss(self,
                    D,
                    cur_level,
                    real,
                    real_mask,
                    obs,
                    obs_mask,
                    syn,
                    cls_real,
                    cls_syn,
                    pixel_cls_real,
                    pixel_cls_syn):
        """Calculate Descriminator loss.

        Args:
            D: discriminator
            cur_level (float32) : progress indicator of
                                  progressive growing network
            real (tensor) : real images
            real_mask (tensor) : domain masks of real images
            obs(tensor) : observed images
            obs_mask (tensor) : domain masks of observed images
            syn (tensor) : synthesized images
            cls_real (tensor) : classes for real images
            cls_syn (tensor) : classes for synthesized images
            pixel_cls_real (tensor) : pixelwise classes for real images
            pixel_cls_syn (tensor) : pixelwise classes for synthesized images

        """
        # adversarial loss
        self.d_losses.d_adver_loss_real = self.calc_adver_loss(cls_real, True)
        self.d_losses.d_adver_loss_syn = self.calc_adver_loss(cls_syn, False)
        self.d_losses.gradient_penalty = 0.0

        if self.gan == Gan.wgan_gp:
            self.d_losses.gradient_penalty = \
                self.calc_gradient_penalty(D,
                                           cur_level,
                                           real,
                                           syn)

        self.d_adver_loss = self.d_losses.d_adver_loss_real +\
            self.d_losses.d_adver_loss_syn +\
            self.d_losses.gradient_penalty

        # pixelwise classification loss
        if pixel_cls_real is None:
            self.d_losses.pixel_loss_real = 0
            self.d_losses.pixel_loss_syn = 0
            self.d_losses.pixel_loss = 0
        else:
            self.d_losses.pixel_loss_real = \
                self.cross_entropy2d(pixel_cls_real, real_mask)
            self.d_losses.pixel_loss_syn = \
                self.cross_entropy2d(pixel_cls_syn, obs_mask)

            self.d_losses.pixel_loss = self.d_losses.pixel_loss_real + \
                self.d_losses.pixel_loss_syn

        self.d_losses.d_loss = self.d_adver_loss +\
            self.lambda_pixel*self.d_losses.pixel_loss

        return self.d_losses


class Vgg16FeatureExtractor(nn.Module):
    """Vgg16FeatureExtractor classes.

    Attributes:
        vgg16_input_size : vgg16 input image size (default = 254)
        features : feature map list of vgg16

    """

    def __init__(self):
        """Class initializer."""
        super(Vgg16FeatureExtractor, self).__init__()

        self.vgg16_input_size = 254
        end_layer = Vgg16Layers.relu4_3
        features = list(vgg16(pretrained=True).features)[:end_layer]

        self.features = nn.ModuleList(features)

    def forward(self, x, extracted_layer=Vgg16Layers.relu2_2):
        """Forward.

        Args:
            x: extracted feature map
            extracted_layer: vgg16 layer number

        """
        if x.shape[2] < self.vgg16_input_size:
            x = self.upsample_Tensor(x, self.vgg16_input_size)
        elif x.shape[2] > self.vgg16_input_size:
            x = self.downsample_Tensor(x, self.vgg16_input_size)

        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == extracted_layer:
                return x
        return x

    def downsample_Tensor(self, x, out_size):
        """Downsample_Tensor.

        Args:
            x: input images
            out_size: down sampled size

        """
        in_size = x.shape[2]

        if in_size < out_size:
            return self.upsample_Tensor(x, out_size)

        kernel_size = in_size // out_size

        if kernel_size == 0:
            return x
        # padding = in_size - out_size*kernel_size
        x = nn.functional.avg_pool2d(x, kernel_size)    # no overlap
        # x = m(x)

        return x

    def upsample_Tensor(self, x, out_size):
        """Upsample Tensor.

        Args:
            x: input images
            out_size: up sampled size

        """
        in_size = x.shape[2]

        if in_size >= out_size:
            return self.downsample_Tensor(x, out_size)

        scale_factor = out_size // in_size

        if (out_size % in_size) != 0:
            scale_factor += 1

        x = nn.functional.upsample(x, scale_factor=scale_factor)

        return x


class MeanFilter(nn.Module):
    """MeanFilter classes.

    Attributes:
        filter : mean filter (convolution module)

    """

    def __init__(self, shape, filter_size):
        """Class initializer."""
        super(MeanFilter, self).__init__()

        self.filter = nn.Conv2d(shape[1],
                                shape[1],
                                filter_size,
                                stride=1,
                                padding=filter_size//2)

        init_weight = 1.0 / (filter_size*filter_size)
        nn.init.constant(self.filter.weight, init_weight)

    def forward(self, x):
        """Forward.

        Args:
            x: input images

        """
        x = self.filter(x)
        return x
