"""Extract CelebA-HQ tfrecord format files."""

import os
import argparse

import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def parse_tfrecord_np(tfr_file):
    """Parse numpy array from tfrecord file.

    Args:
        tfr_file: tfrecord file path.

    Return: tuple, (array, array)
    """
    ex = tf.train.Example()
    ex.ParseFromString(tfr_file)
    shape = ex.features.feature['shape'].int64_list.value
    data = ex.features.feature['data'].bytes_list.value[0]
    attr = ex.features.feature['attr'].bytes_list.value[0]
    return np.fromstring(data, np.uint8).reshape(shape), \
        np.fromstring(attr, np.uint8)


def extract_tfrecord_png(tfr_files, save_dir):
    """Extract png image files from tfrecord.

    Args:
        tfr_file: tfrecord file path.
        save_dir: directory path for saving images.
    """
    for tfr_file in tfr_files:
        tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE) # noqa E501
        i = 0
        for record in tf.python_io.tf_record_iterator(tfr_file, tfr_opt):
            img, attr = parse_tfrecord_np(record)
            img = img.transpose(1, 2, 0)
            res = str(img.shape[1])
            img_file_name = ''.join([str(i) for i in attr]) + f'_{i}.png'
            img_file_path = os.path.join(save_dir, res, img_file_name)
            plt.imsave(img_file_path, img)
            i += 1


def make_sub_dirs(save_dir):
    """Make directories for saving files.

    Args:
        save_dir: parent directory path.
    """
    res = [2**i for i in range(2, 11)]
    for i in res:
        dir_path = os.path.join(save_dir, str(i))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def load_tfrecord_from_dir(tfr_dir):
    """Load tfrecord files from directory.

    Args:
        tfr_dir: directory path having tfrecord files.
    """
    return sorted(glob.glob(tfr_dir + '/tfrecord*'))


if __name__ == '__main__':
    print("Start")
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfr_dir',
                        default='../test_data/celebA/tfrecord/',
                        help='Directory containing tfrecord files')
    parser.add_argument('--save_dir',
                        default='../test_data/celebA/tfrecord/',
                        help='Directory for save png files')
    args = parser.parse_args()

    print("Make Dirs")
    make_sub_dirs(args.save_dir)
    tfr_files = load_tfrecord_from_dir(args.tfr_dir)
    extract_tfrecord_png(tfr_files, args.save_dir)
    print("End")
