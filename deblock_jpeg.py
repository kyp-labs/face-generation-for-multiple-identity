# !/usr/bin/env python3
"""                          _              _
  _ __   ___ _   _ _ __ __ _| |   ___ _ __ | |__   __ _ _ __   ___ ___
 | '_ \ / _ \ | | | '__/ _` | |  / _ \ '_ \| '_ \ / _` | '_ \ / __/ _ \\
 | | | |  __/ |_| | | | (_| | | |  __/ | | | | | | (_| | | | | (_|  __/
 |_| |_|\___|\__,_|_|  \__,_|_|  \___|_| |_|_| |_|\__,_|_| |_|\___\___|

This code is modified from https://github.com/alexjc/neural-enhance
"""
#
# Copyright (c) 2016, Alex J. Champandard.
#
# Neural Enhance is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License version 3.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

# flake8: noqa
__version__ = '0.3'

import os
import sys
import bz2
import pickle
import argparse
import itertools
import collections
import pandas as pd
#from thread_pool import ThreadPool

TEST_LANDMARKS_PATH = './dataset/VGGFACE2/bb_landmark/loose_landmark_test.csv'
TEST_IMAGES_PATH = './dataset/VGGFACE2/test'
TRAIN_LANDMARKS_PATH = './dataset/VGGFACE2/bb_landmark/loose_landmark_train.csv'
TRAIN_IMAGES_PATH = './dataset/VGGFACE2/train'
IDENTITY_INFO_PATH = './identity_info.csv'
IMAGE_OUTPUT_PATH = './deblocked'

# Configure all options first so we can later custom-load other libraries (Theano) based on device specified by user.
parser = argparse.ArgumentParser(description='Generate a new image by applying style onto a content image.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add_arg = parser.add_argument
add_arg('--identity-info',      default=IDENTITY_INFO_PATH, type=str,    help='Path of identity_info.csv')
add_arg('--num-threads',        default=4, type=int,                     help='Number of concurrent threads (default: 4)')
add_arg('--num-tasks',          default=100, type=int,                   help='Number of concurrent processing tasks (default: 100)')
add_arg('--zoom',               default=1, type=int,                     help='Resolution increase factor for inference.')
add_arg('--rendering-tile',     default=80, type=int,                    help='Size of tiles used for rendering images.')
add_arg('--rendering-overlap',  default=24, type=int,                    help='Number of pixels padding around each tile.')
add_arg('--generator-upscale',  default=2, type=int,                     help='Steps of 2x up-sampling as post-process.')
add_arg('--generator-downscale',default=0, type=int,                     help='Steps of 2x down-sampling as preprocess.')
add_arg('--generator-filters',  default=[64], nargs='+', type=int,       help='Number of convolution units in network.')
add_arg('--generator-blocks',   default=4, type=int,                     help='Number of residual blocks per iteration.')
add_arg('--generator-residual', default=2, type=int,                     help='Number of layers in a residual block.')
add_arg('--device',             default='cuda', type=str,                 help='Name of the CPU/GPU to use, for Theano.')
args = parser.parse_args()


#----------------------------------------------------------------------------------------------------------------------

def error(message, *lines):
    string = "\nERROR: " + message + "\n" + "\n".join(lines) + "\n" if lines else ""
    print(string)
    sys.exit(-1)


def extend(lst): return itertools.chain(lst, itertools.repeat(lst[-1]))

print("""    {}Super Resolution for images and videos powered by Deep Learning!\
  - Code licensed as AGPLv3, models under CC BY-NC-SA.""".format(__doc__))

# Load the underlying deep learning libraries based on the device specified.  If you specify THEANO_FLAGS manually,
# the code assumes you know what you are doing and they are not overriden!
os.environ.setdefault('THEANO_FLAGS', 'floatX=float32,device={},force_device=True,allow_gc=True,'\
                                      'print_active_device=False,gpuarray.single_stream={}'.format(args.device, False))

# Scientific & Imaging Libraries
import numpy as np
import scipy.ndimage, scipy.misc

# Numeric Computing (GPU)
import theano, theano.tensor as T
T.nnet.softminus = lambda x: x - T.nnet.softplus(x)

# Deep Learning Framework
import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import InputLayer, ElemwiseSumLayer

print('  - Using the device `{}` for neural computation.\n'.format(theano.config.device))


#======================================================================================================================
# Convolution Networks
#======================================================================================================================

class SubpixelReshuffleLayer(lasagne.layers.Layer):
    """Based on the code by ajbrock: https://github.com/ajbrock/Neural-Photo-Editor/
    """

    def __init__(self, incoming, channels, upscale, **kwargs):
        super(SubpixelReshuffleLayer, self).__init__(incoming, **kwargs)
        self.upscale = upscale
        self.channels = channels

    def get_output_shape_for(self, input_shape):
        def up(d): return self.upscale * d if d else d
        return (input_shape[0], self.channels, up(input_shape[2]), up(input_shape[3]))

    def get_output_for(self, input, deterministic=False, **kwargs):
        out, r = T.zeros(self.get_output_shape_for(input.shape)), self.upscale
        for y, x in itertools.product(range(r), repeat=2):
            out=T.inc_subtensor(out[:,:,y::r,x::r], input[:,r*y+x::r*r,:,:])
        return out


class Model(object):

    def __init__(self):
        self.network = collections.OrderedDict()
        self.network['img'] = InputLayer((None, 3, None, None))
        self.network['seed'] = InputLayer((None, 3, None, None))

        config, params = self.load_model()
        self.setup_generator(self.last_layer(), config)
        self.load_generator(params)
        self.compile()

    #------------------------------------------------------------------------------------------------------------------
    # Network Configuration
    #------------------------------------------------------------------------------------------------------------------

    def last_layer(self):
        return list(self.network.values())[-1]

    def make_layer(self, name, input, units, filter_size=(3,3), stride=(1,1), pad=(1,1), alpha=0.25):
        conv = ConvLayer(input, units, filter_size, stride=stride, pad=pad, nonlinearity=None)
        prelu = lasagne.layers.ParametricRectifierLayer(conv, alpha=lasagne.init.Constant(alpha))
        self.network[name+'x'] = conv
        self.network[name+'>'] = prelu
        return prelu

    def make_block(self, name, input, units):
        self.make_layer(name+'-A', input, units, alpha=0.1)
        # self.make_layer(name+'-B', self.last_layer(), units, alpha=1.0)
        return ElemwiseSumLayer([input, self.last_layer()]) if args.generator_residual else self.last_layer()

    def setup_generator(self, input, config):
        for k, v in config.items(): setattr(args, k, v)
        args.zoom = 2**(args.generator_upscale - args.generator_downscale)

        units_iter = extend(args.generator_filters)
        units = next(units_iter)
        self.make_layer('iter.0', input, units, filter_size=(7,7), pad=(3,3))

        for i in range(0, args.generator_downscale):
            self.make_layer('downscale%i'%i, self.last_layer(), next(units_iter), filter_size=(4,4), stride=(2,2))

        units = next(units_iter)
        for i in range(0, args.generator_blocks):
            self.make_block('iter.%i'%(i+1), self.last_layer(), units)

        for i in range(0, args.generator_upscale):
            u = next(units_iter)
            self.make_layer('upscale%i.2'%i, self.last_layer(), u*4)
            self.network['upscale%i.1'%i] = SubpixelReshuffleLayer(self.last_layer(), u, 2)

        self.network['out'] = ConvLayer(self.last_layer(), 3, filter_size=(7,7), pad=(3,3), nonlinearity=None)

    #------------------------------------------------------------------------------------------------------------------
    # Input / Output
    #------------------------------------------------------------------------------------------------------------------

    def list_generator_layers(self):
        for l in lasagne.layers.get_all_layers(self.network['out'], treat_as_input=[self.network['img']]):
            if not l.get_params(): continue
            name = list(self.network.keys())[list(self.network.values()).index(l)]
            yield (name, l)

    def get_filename(self, absolute=False):
        filename = 'ne%ix-%s-%s-%s.pkl.bz2' % (args.zoom, 'photo', 'repair', __version__)
        return os.path.join(os.path.dirname(__file__), filename) if absolute else filename

    def load_model(self):
        if not os.path.exists(self.get_filename(absolute=True)):
            error("Model file with pre-trained convolution layers not found. Download it here...",
                  "https://github.com/alexjc/neural-enhance/releases/download/v%s/%s"%(__version__, self.get_filename()))
        print('  - Loaded file `{}` with trained model.'.format(self.get_filename()))
        return pickle.load(bz2.open(self.get_filename(absolute=True), 'rb'))

    def load_generator(self, params):
        if len(params) == 0: return
        for k, l in self.list_generator_layers():
            assert k in params, "Couldn't find layer `%s` in loaded model.'" % k
            assert len(l.get_params()) == len(params[k]), "Mismatch in types of layers."
            for p, v in zip(l.get_params(), params[k]):
                assert v.shape == p.get_value().shape, "Mismatch in number of parameters for layer {}.".format(k)
                p.set_value(v.astype(np.float32))

    def compile(self):
        # Helper function for rendering test images during training, or standalone inference mode.
        input_tensor, seed_tensor = T.tensor4(), T.tensor4()
        input_layers = {self.network['img']: input_tensor, self.network['seed']: seed_tensor}
        output = lasagne.layers.get_output([self.network[k] for k in ['seed','out']], input_layers, deterministic=True)
        self.predict = theano.function([seed_tensor], output)


def deblock_jpeg(model, loose_landmarks, img_dir, start_idx, end_idx, num_threads=4, num_tasks=100):
    def process_func(idx):
        predict = model.predict.copy()
        img_name = loose_landmarks[0][idx]
        img = scipy.ndimage.imread(img_dir+'/'+img_name+'.jpg', mode='RGB')
        print('->     processing image: ', img_name, '...')

        # Snap the image to a shape that's compatible with the generator (2x, 4x)
        s = 2 ** max(args.generator_upscale, args.generator_downscale)
        by, bx = img.shape[0] % s, img.shape[1] % s
        img = img[by-by//2:img.shape[0]-by//2,bx-bx//2:img.shape[1]-bx//2,:]

        # Prepare paded input image as well as output buffer of zoomed size.
        s, p, z = args.rendering_tile, args.rendering_overlap, args.zoom
        image = np.pad(img, ((p, p), (p, p), (0, 0)), mode='reflect')
        output = np.zeros((img.shape[0] * z, img.shape[1] * z, 3), dtype=np.float32)

        # Iterate through the tile coordinates and pass them through the network.
        for y, x in itertools.product(range(0, img.shape[0], s), range(0, img.shape[1], s)):
            img = np.transpose(image[y:y+p*2+s,x:x+p*2+s,:] / 255.0 - 0.5, (2, 0, 1))[np.newaxis].astype(np.float32)
            *_, repro = predict(img)
            output[y*z:(y+s)*z,x*z:(x+s)*z,:] = np.transpose(repro[0] + 0.5, (1, 2, 0))[p*z:-p*z,p*z:-p*z,:]
        output = output.clip(0.0, 1.0) * 255.0

        return (scipy.misc.toimage(output, cmin=0, cmax=255), img_name)

    # Single threading execution code
    for idx in range(start_idx, end_idx+1):
        img, img_name = process_func(idx)
        img_name = img_name.replace('/', '-')
        img.save(IMAGE_OUTPUT_PATH+'/'+img_name+'.PNG', 'PNG')
        print('->     saved ', img_name, '...')

#    # Multithreading execution code
#    with ThreadPool(args.num_threads) as pool:
#        for img, img_name in pool.process_items_concurrently(
#                                                range(start_idx, end_idx+1),
#                                                process_func=process_func,
#                                                max_items_in_flight=num_tasks):
#            img.save(img_dir+'/'+img_name+'_deblocked.PNG', 'PNG')
#            print('saved ', img_name, '...')


if __name__ == "__main__":
    assert(args.num_threads > 0 and args.num_tasks > 0)

    model = Model()

    assert(os.path.exists(args.identity_info))
    identity_info = pd.read_csv(args.identity_info)

    if not os.path.exists(IMAGE_OUTPUT_PATH):
        os.makedirs(IMAGE_OUTPUT_PATH)


    assert(os.path.exists(TEST_LANDMARKS_PATH))
    assert(os.path.exists(TRAIN_LANDMARKS_PATH))
    print('loading landmarks.csv...')
    loose_landmarks_test = pd.read_csv(TEST_LANDMARKS_PATH,
                                       header=None,
                                       low_memory=False)
    loose_landmarks_train = pd.read_csv(TRAIN_LANDMARKS_PATH,
                                        header=None,
                                        low_memory=False)

    assert(os.path.exists(TEST_IMAGES_PATH))
    assert(os.path.exists(TRAIN_IMAGES_PATH))

    for i in range(len(identity_info['Class_ID'])):
        class_id = identity_info['Class_ID'][i]
        name = identity_info['Name'][i]
        print('processing identity: ', class_id, name, '...')

        is_train = identity_info['Flag'][i]
        start_idx = identity_info['Start_Idx'][i] + 1
        end_idx = start_idx + identity_info['Sample_Num'][i] - 1

        loose_landmarks = loose_landmarks_train if is_train else loose_landmarks_test
        img_dir = TRAIN_IMAGES_PATH if is_train else TEST_IMAGES_PATH

        deblock_jpeg(model, loose_landmarks, img_dir, start_idx, end_idx,
                     args.num_threads, args.num_tasks)
