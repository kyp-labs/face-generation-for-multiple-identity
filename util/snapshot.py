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
import matplotlib.pyplot as plt

import config
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
        g_loss_hist : loss history for generator (for plotting)
        d_loss_hist : loss history for discriminator (for plotting)
        real : real images
        syn : synthesized images

    """

    def __init__(self, use_coda):
        """Class initializer."""
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
        restore_dir = config.checkpoint.restore_dir
        which_file = config.checkpoint.which_file  # 128x128-transition-105000

        if config.checkpoint.restore is False \
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
        self.exp_dir = config.snapshot.exp_dir
        self.sample_dir = \
            os.path.join(self.exp_dir, self.current_time, 'samples')
        self.ckpt_dir = os.path.join(self.exp_dir, self.current_time, 'ckpts')

        os.makedirs(self.sample_dir)
        os.makedirs(self.ckpt_dir)

    def prepare_logging(self):
        """Prepare_logging."""
        root_log_dir = config.logging.log_dir
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
                 d_losses,
                 g_loss_hist,
                 d_loss_hist):
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
            g_loss_hist : loss history for generator (for plotting)
            d_loss_hist : loss history for discriminator (for plotting)

        """
        self.g_losses = g_losses
        self.d_losses = d_losses
        self.g_loss_hist = g_loss_hist
        self.d_loss_hist = d_loss_hist
        self.real = real
        self.syn = syn

        # ===report ===
        self.line_summary(global_it, it, total_it, phase, cur_resol, cur_level)

        if config.snapshot.enable_threading:
            t = threading.Thread(target=self._snapshot,
                                 args=(global_it,
                                       it,
                                       total_it,
                                       phase,
                                       cur_resol,
                                       cur_level,
                                       minibatch_size,
                                       G,
                                       D,
                                       optim_G,
                                       optim_D))
            t.start()
        else:
            self._snapshot(global_it,
                           it,
                           total_it,
                           phase,
                           cur_resol,
                           cur_level,
                           minibatch_size,
                           G,
                           D,
                           optim_G,
                           optim_D)

    def _snapshot(self,
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
        sample_freq = \
            config.snapshot.sample_freq_dict.get(cur_resol,
                                                 config.snapshot.sample_freq)
        save_freq = \
            config.checkpoint.save_freq_dict.get(cur_resol,
                                                 config.checkpoint.save_freq)
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

            if config.snapshot.draw_plot:
                filename = '%s-%dx%d-%s-%s-loss.png' % (str(global_it).
                                                        zfill(6),
                                                        cur_resol,
                                                        cur_resol,
                                                        str(it).zfill(6),
                                                        phase)
                self.plot_loss(global_it,
                               os.path.join(self.sample_dir, filename),
                               False)

        # ===tensorboard visualization===
        # if (it % sample_freq == 0) or it == total_it:
        #    self.tensorboard(global_it, it, phase, cur_resol, G, D)

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
        formation = '%d [%dx%d](%d/%d)%.1f %s " + \
                    "| G:%.3f, D:%.3f " + \
                    "| G_adv:%.3f, R:%.3f, F:%.3f, B:%.3f " + \
                    "| D_adv:%.3f(%.3f,%.3f), A:%.3f, GP:%.3f'
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
                  self.d_losses.d_adver_loss,
                  self.d_losses.d_adver_loss_real,
                  self.d_losses.d_adver_loss_syn,
                  self.d_losses.att_loss,
                  self.d_losses.gradient_penalty)

        print(formation % values)

    def image_sampling(self, minibatch_size):
        """Image_sampling.

        Args:
            minibatch_size: minibatch size

        """
        n_row = config.snapshot.rows_map[minibatch_size]
        if n_row >= minibatch_size:
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

    def tensorboard(self, global_it, it, phase, cur_resol, G, D):
        """Tensorboard.

        Args:
            global_it : global # of iterations through training
            it: current # of iterations in the phases of the layer
            phase: training, transition, replaying
            cur_resol: image resolution of current layer
            G: generator
            D: discriminator

        """
        # (1) Log the scalar values
        prefix = str(global_it)+'/' + str(cur_resol)+'/' + str(phase) + '/'
        # prefix = str(cur_resol)+'/' + str(phase) + '/'

        info = {prefix + 'G_loss': self.g_losses.g_loss,
                prefix + 'G_adver_loss': self.g_losses.g_adver_loss,
                prefix + 'recon_loss': self.g_losses.recon_loss,
                prefix + 'feat_loss': self.g_losses.feat_loss,
                prefix + 'bdy_loss': self.g_losses.bdy_loss,
                prefix + 'D_loss': self.d_losses.d_loss,
                prefix + 'D_adver_loss': self.d_losses.d_adver_loss,
                prefix + 'D_adver_loss_syn': self.d_losses.d_adver_loss_syn,
                prefix + 'D_adver_loss_real': self.d_losses.d_adver_loss_real,
                prefix + 'att_loss': self.d_losses.att_loss,
                prefix + 'gradient_penalty': self.d_losses.gradient_penalty}

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, global_it)

        # (2) Log values and gradients of the parameters (histogram)
        for tag, value in G.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary('G/' + prefix + tag,
                                      util.tensor2numpy(self.use_cuda, value),
                                      it)
            if value.grad is not None:
                self.logger.histo_summary('G/' + prefix + tag + '/grad',
                                          util.tensor2numpy(self.use_cuda,
                                                            value.grad),
                                          it)

        for tag, value in D.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary('D/' + prefix + tag,
                                      util.tensor2numpy(self.use_cuda, value),
                                      it)
            if value.grad is not None:
                self.logger.histo_summary('D/' + prefix + tag + '/grad',
                                          util.tensor2numpy(self.use_cuda,
                                                            value.grad),
                                          it)

    # Plot losses
    def plot_loss(self, global_it, file_name, show=False):
        """Plot loss.

        Args:
            global_it : global # of iterations through training
            file_name: file name for saving a plot image
            show: flag to show plot on screen

        """
        if config.snapshot.draw_plot is False:
            return

        g_loss_hist = self.g_loss_hist.g_loss_hist
        d_loss_hist = self.d_loss_hist.d_loss_hist

        fig, ax = plt.subplots()
        ax.set_xlim(0, len(g_loss_hist))
        ax.set_ylim(0, max(np.max(g_loss_hist), np.max(d_loss_hist))*1.1)
        plt.xlabel('Iteration {0}'.format(global_it + 1))
        plt.ylabel('Loss values')
        plt.plot(g_loss_hist, label='Generator')
        plt.plot(d_loss_hist, label='Discriminator')
        plt.legend()

        plt.savefig(file_name)

        if show:
            plt.show()
        else:
            plt.close()
