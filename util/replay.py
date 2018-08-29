"""replay.py.

This module includes ReplayMemory classe
which matains old input data to replay discriminator with them.

"""
import torch
import random as ran
import config


class ReplayMemory():
    """ReplayMemory classes.

    Attributes:
        use_cuda : flag for cuda use
        enabled : flag for replay mode enablement
        replay_memory : maintain [real, attr_real, mask, obs, attr_obs, syn]
        cur_resol : resolution of replay data
        max_memory_size : maximum replay memory size

    """

    def __init__(self, use_cuda, enabled):
        """Init variables."""
        self.use_cuda = use_cuda
        self.enabled = enabled
        self.replay_memory = []
        self.cur_resol = 0
        dict = config.replay.max_memory_size_dict
        self.max_memory_size = dict.get(self.cur_resol,
                                        config.replay.max_memory_size)

    def reset(self, cur_resol):
        """Reset replay memory for [cur_resol] resolution.

        Args:
            cur_resol: resolution of replay data

        """
        assert cur_resol >= 1

        if self.enabled is False:
            return

        self.cur_resol = cur_resol
        self.replay_memory.clear()

        dict = config.replay.max_memory_size_dict
        self.max_memory_size = dict.get(self.cur_resol,
                                        config.replay.max_memory_size)

    def append(self, cur_resol, real, mask, obs, syn):
        """Append new data to replay memory.

        Args:
            cur_resol: resolution of replay data
            real: real iag
            mask : binary mask
            obs: observed images
            syn:synthesized images

        """
        assert cur_resol >= 1
        assert cur_resol == self.cur_resol

        if self.enabled is False:
            return

        mem_len = len(self.replay_memory)
        over_num = mem_len + real.shape[0] - self.max_memory_size
        self.delete(over_num)

        for i in range(real.shape[0]):
            self.replay_memory.append([real[i],
                                       mask[i],
                                       obs[i],
                                       syn[i]])

    def delete(self, del_len):
        """Delete items from start to [del_len]th items of replay memory.

        Args:
            del_len: number of items to be deleted

        """
        if self.enabled is False:
            return

        if del_len <= 0:
            return

        del self.replay_memory[:del_len]

    def get_batch(self, cur_resol, batch_size):
        """Get batch from replay memroy at random.

        Args:
            cur_resol: resolution of replay data
            batch_size: batch size

        """
        assert cur_resol == self.cur_resol
        assert batch_size > 0

        if self.enabled is False:
            return None, None, None, None, None, None

        if len(self.replay_memory) == 0:
            return None, None, None, None, None, None

        replay_l = [replay_item for replay_item
                    in ran.sample(self.replay_memory, batch_size)]

        real = torch.stack([replay_item[0] for replay_item in replay_l])
        mask = torch.stack([replay_item[1] for replay_item in replay_l])
        obs = torch.stack([replay_item[2] for replay_item in replay_l])
        syn = torch.stack([replay_item[3] for replay_item in replay_l])

        return real, mask, obs, syn
