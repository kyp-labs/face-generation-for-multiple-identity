"""snapshot.py.

This module includes Snapshot class
which makes snapshots of intermediate images, checkpoints and tensorboard log

"""
import os
import time
import numpy as np
from scipy.misc import imsave
import torch
import threading

import util.util as util
from util.logger import Logger
from util.util import Phase


class Snapshot(object):
    """Snapshot classes.

    Attributes:
        use_cuda : flag for cuda use
        current_time : current system time
        is_restored : flag for whether checkpoint file is restored or not
        _global_it : global # of iterations restored
        _resolution : resolution restored
        _phase : phase restored
        _it : # of iterations restored
        exp_dir : export directory
        time : restored time
        sample_dir : smaple snapshot directory
        ckpt_dir : checkpoint directory
        log_dir : log directory
        logger : logger
        g_losses : losses of generator
        d_losses : losses of discriminator
        real : real images
        syn : synthesized images

    """

    def __init__(self, config, use_coda):
        """Class initializer."""
        self.config = config
        self.use_cuda = use_coda
        self.current_time = time.strftime('%Y-%m-%d %H%M%S')
        self.is_restored = False

    def restore_model(self, G, D, optim_G, optim_D):
        """Restore model from checkpoint.

        Args:
            G: generator
            D: discriminator
            optim_G: optimizer of generator
            optim_D: optimizer of discriminator

        """
        restore_dir = self.config.checkpoint.restore_dir
        # 128x128-transition-105000
        which_file = self.config.checkpoint.which_file

        if self.config.checkpoint.restore is False \
           or restore_dir == "" or which_file == "":
            self.is_restored = False
            self.new_directory()
            return

        self.is_restored = True

        pattern = which_file.split('-')
        self._global_it = int(pattern[0])
        self._resolution = int(pattern[1].split('x')[1])
        self._phase = Phase.training if pattern[1] == "training" \
            else Phase.transition
        self._it = int(pattern[3])

        tmp = restore_dir.split('/')
        self.exp_dir = '/'.join(tmp[:-1])
        self.time = tmp[-1]

        self.sample_dir = os.path.join(restore_dir, 'samples')
        self.ckpt_dir = os.path.join(restore_dir, 'ckpts')
        assert os.path.exists(self.sample_dir) \
            and os.path.exists(self.ckpt_dir)

        filename = os.path.join(self.ckpt_dir, which_file)
        checkpoint = torch.load(filename)

        G.load_state_dict(checkpoint["G"])
        D.load_state_dict(checkpoint["D"])
        optim_G.load_state_dict(checkpoint["optim_G"])
        optim_D.load_state_dict(checkpoint["optim_D"])

        print('Restored from dir: %s, pattern: %s' %
              (self.exp_dir, which_file))

    def save_model(self, file_name, G, D, optim_G, optim_D):
        """Save_model.

        Args:
            file_name: checkpoint file name
            G: generator
            D: discriminator
            optim_G: optimizer of generator
            optim_D: optimizer of discriminator

        """
        checkpoint = {
            'G': G.state_dict(),
            'D': D.state_dict(),
            'optim_G': optim_G.state_dict(),
            'optim_D': optim_D.state_dict()
        }
        torch.save(checkpoint, file_name)

    def new_directory(self):
        """New_directory."""
        self.exp_dir = self.config.snapshot.exp_dir
        self.sample_dir = \
            os.path.join(self.exp_dir, self.current_time, 'samples')
        self.ckpt_dir = os.path.join(self.exp_dir, self.current_time, 'ckpts')

        os.makedirs(self.sample_dir)
        os.makedirs(self.ckpt_dir)

    def prepare_logging(self):
        """Prepare_logging."""
        root_log_dir = self.config.logging.log_dir
        if os.path.exists(root_log_dir) is False:
            os.makedirs(root_log_dir)

        self.log_dir = os.path.join(root_log_dir, self.current_time, "")
        self.logger = Logger(self.log_dir)

    def snapshot(self,
                 global_it,
                 it,
                 total_it,
                 phase,
                 cur_resol,
                 cur_level,
                 minibatch_size,
                 real,
                 syn,
                 G,
                 D,
                 optim_G,
                 optim_D,
                 g_losses,
                 d_losses):
        """Snapshot.

        Args:
            global_it : global # of iterations through training
            it: current # of iterations in the phases of the layer
            total_it: total # of iterations in the phases of the layer
            phase: training, transition, replaying
            cur_resol: image resolution of current layer
            cur_level: progress indicator of progressive growing network
            minibatch_size: minibatch size
            real: real images
            syn: synthesized images
            G: generator
            D: discriminator
            optim_G: optimizer of generator
            optim_D: optimizer of discriminator
            g_losses : losses of generator
            d_losses : losses of discriminator

        """
        self.g_losses = g_losses
        self.d_losses = d_losses
        self.real = real
        self.syn = syn

        # ===report ===
        self.line_summary(global_it, it, total_it, phase, cur_resol, cur_level)
        self.log_loss_to_tensorboard(global_it)

        args = (global_it, it, total_it, phase, cur_resol, cur_level,
                minibatch_size, G, D, optim_G, optim_D)

        if self.config.snapshot.enable_threading:
            t = threading.Thread(target=self.periodic_snapshot, args=args)
            t.start()
        else:
            self.periodic_snapshot(*args)

    def periodic_snapshot(self,
                          global_it,
                          it,
                          total_it,
                          phase,
                          cur_resol,
                          cur_level,
                          minibatch_size,
                          G,
                          D,
                          optim_G,
                          optim_D):
        """Snapshot.

        Args:
            global_it : global # of iterations through training
            it: current # of iterations in the phases of the layer
            total_it: total # of iterations in the phases of the layer
            phase: training, transition, replaying
            cur_resol: image resolution of current layer
            cur_level: progress indicator of progressive growing network
            minibatch_size: minibatch size
            G: generator
            D: discriminator
            optim_G: optimizer of generator
            optim_D: optimizer of discriminator

        """
        sample_freq_dict = self.config.snapshot.sample_freq_dict
        sample_freq = sample_freq_dict.get(cur_resol,
                                           self.config.snapshot.sample_freq)
        save_freq_dict = self.config.checkpoint.save_freq_dict
        save_freq = save_freq_dict.get(cur_resol,
                                       self.config.checkpoint.save_freq)
        # ===generate sample images===
        samples = []
        if (it % sample_freq == 1) or it == total_it:
            samples = self.image_sampling(minibatch_size)
            filename = '%s-%dx%d-%s-%s.png' % (str(global_it).zfill(6),
                                               cur_resol,
                                               cur_resol,
                                               str(it).zfill(6),
                                               phase)
            imsave(os.path.join(self.sample_dir, filename), samples)

        # ===tensorboard visualization===
        if (it % sample_freq == 1) or it == total_it:
            self.log_weight_to_tensorboard(global_it, G, D)

        # ===save model===
        if (it % save_freq == 1) or it == total_it:
            filename = '%s-%dx%d-%s-%s.pth' % (str(global_it).zfill(6),
                                               cur_resol,
                                               cur_resol,
                                               phase,
                                               str(it).zfill(6))
            self.save_model(os.path.join(self.ckpt_dir, filename),
                            G, D, optim_G, optim_D)

    def line_summary(self,
                     global_it,
                     it,
                     total_it,
                     phase,
                     cur_resol,
                     cur_level):
        """Line_summary.

        Args:
            global_it : global # of iterations through training
            it: current # of iterations in the phases of the layer
            total_it: total # of iterations in the phases of the layer
            phase: training, transition, replaying
            cur_resol: image resolution of current layer
            cur_level: progress indicator of progressive growing network

        """
        formation = '%d [%dx%d](%d/%d)%.1f %s ' +\
                    '| G:%.3f, D:%.3f ' +\
                    '| G_adv:%.3f, R:%.3f, F:%.3f, B:%.3f, P:%.3f, C: %.3f,' +\
                    '| D_adv:%.3f(%.3f,%.3f), GP:%.3f' +\
                    '| Pixel:%.3f(%.3f,%.3f)'
        values = (global_it,
                  cur_resol,
                  cur_resol,
                  it,
                  total_it,
                  cur_level, phase,
                  self.g_losses.g_loss,
                  self.d_losses.d_loss,
                  self.g_losses.g_adver_loss,
                  self.g_losses.recon_loss,
                  self.g_losses.feat_loss,
                  self.g_losses.bdy_loss,
                  self.g_losses.pixel_loss,
                  self.g_losses.cycle_loss,
                  self.d_losses.d_adver_loss,
                  self.d_losses.d_adver_loss_real,
                  self.d_losses.d_adver_loss_syn,
                  self.d_losses.gradient_penalty,
                  self.d_losses.pixel_loss,
                  self.d_losses.pixel_loss_real,
                  self.d_losses.pixel_loss_syn)

        print(formation % values)

    def image_sampling(self, minibatch_size):
        """Image_sampling.

        Args:
            minibatch_size: minibatch size

        """
        n_row = self.config.snapshot.rows_map[minibatch_size]
        if n_row > minibatch_size:
            n_row = minibatch_size // 2
        n_col = int(np.ceil(minibatch_size / float(n_row)))

        # sample_idx = np.random.randint(minibatch_size, size=n_row*n_col)
        samples = []
        i = j = 0
        for _ in range(n_row):
            one_row = []
            # syn
            for _ in range(n_col):
                one_row.append(self.syn[i].cpu().data.numpy())
                i += 1
            # real
            for _ in range(n_col):
                one_row.append(self.real[j].cpu().data.numpy())
                j += 1
            samples += [np.concatenate(one_row, axis=2)]
        samples = np.concatenate(samples, axis=1).transpose([1, 2, 0])

        half = samples.shape[1] // 2
        samples[:, :half, :] = \
            samples[:, :half, :] - np.min(samples[:, :half, :])
        samples[:, :half, :] = \
            samples[:, :half, :] / np.max(samples[:, :half, :])
        samples[:, half:, :] = \
            samples[:, half:, :] - np.min(samples[:, half:, :])
        samples[:, half:, :] = \
            samples[:, half:, :] / np.max(samples[:, half:, :])

        return samples

    def log_loss_to_tensorboard(self, global_it):
        """Tensorboard.

        Args:
            global_it : global # of iterations through training

        """
        # (1) Log the scalar values
        info = {'Generator/Loss':
                self.g_losses.g_loss,
                'Generator/Adversarial Loss':
                self.g_losses.g_adver_loss,
                'Generator/Reconstruction Loss':
                self.g_losses.recon_loss,
                'Generator/Feature Loss':
                self.g_losses.feat_loss,
                'Generator/Boundary Loss':
                self.g_losses.bdy_loss,
                'Generator/Pixelwise Classfication Loss':
                self.g_losses.pixel_loss,
                'Generator/Cycle Consistency Loss':
                self.g_losses.cycle_loss,
                'Discriminator/Loss':
                self.d_losses.d_loss,
                'Discriminator/Adversarial Loss':
                self.d_losses.d_adver_loss,
                'Discriminator/Adversarial Loss (R)':
                self.d_losses.d_adver_loss_real,
                'Discriminator/Adversarial Loss (S)':
                self.d_losses.d_adver_loss_syn,
                'Discriminator/Gradient Penalty':
                self.d_losses.gradient_penalty,
                'Discriminator/Pixelwise Classfication Loss':
                self.d_losses.pixel_loss,
                'Discriminator/Pixelwise Classfication Loss (R)':
                self.d_losses.pixel_loss_real,
                'Discriminator/Pixelwise Classfication Loss (S)':
                self.d_losses.pixel_loss_syn}

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, global_it)

    def log_weight_to_tensorboard(self, global_it, G, D):
        """Tensorboard.

        Args:
            global_it : global # of iterations through training
            G: generator
            D: discriminator

        """
        # (2) Log values and gradients of the parameters (histogram)
        for tag, value in G.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary('Generator/' + tag,
                                      util.tensor2numpy(self.use_cuda, value),
                                      global_it)
            if value.grad is not None:
                self.logger.histo_summary('Generator/' + tag + '/grad',
                                          util.tensor2numpy(self.use_cuda,
                                                            value.grad),
                                          global_it)

        for tag, value in D.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary('Discriminator/' + tag,
                                      util.tensor2numpy(self.use_cuda, value),
                                      global_it)
            if value.grad is not None:
                self.logger.histo_summary('Discriminator/' + tag + '/grad',
                                          util.tensor2numpy(self.use_cuda,
                                                            value.grad),
                                          global_it)
