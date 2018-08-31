"""Test training."""

import config
import numpy as np
from train import FaceGen

if __name__ == "__main__":
    np.random.seed(config.common.random_seed)

    print('Running FaceGen()...')
    facegen = FaceGen()
    facegen.train()

    print('Exiting...')
