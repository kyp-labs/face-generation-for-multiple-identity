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


# Configure all options first so we can later custom-load other libraries (Theano) based on device specified by user.
parser = argparse.ArgumentParser(description='Generate a new image by applying style onto a content image.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add_arg = parser.add_argument
add_arg('files', nargs='*', default=[])
add_arg('--zoom',               default=1, type=int,                help='Resolution increase factor for inference.')
add_arg('--rendering-tile',     default=80, type=int,               help='Size of tiles used for rendering images.')
add_arg('--rendering-overlap',  default=24, type=int,               help='Number of pixels padding around each tile.')
add_arg('--rendering-histogram',default=False, action='store_true', help='Match color histogram of output to input.')
add_arg('--type',               default='photo', type=str,          help='Name of the neural network to load/save.')
add_arg('--model',              default='repair', type=str,         help='Specific trained version of the model.')
add_arg('--batch-shape',        default=192, type=int,              help='Resolution of images in training batch.')
add_arg('--batch-size',         default=15, type=int,               help='Number of images per training batch.')
add_arg('--generator-upscale',  default=2, type=int,                help='Steps of 2x up-sampling as post-process.')
add_arg('--generator-downscale',default=0, type=int,                help='Steps of 2x down-sampling as preprocess.')
add_arg('--generator-filters',  default=[64], nargs='+', type=int,  help='Number of convolution units in network.')
add_arg('--generator-blocks',   default=4, type=int,                help='Number of residual blocks per iteration.')
add_arg('--generator-residual', default=2, type=int,                help='Number of layers in a residual block.')
add_arg('--perceptual-layer',   default='conv2_2', type=str,        help='Which VGG layer to use as loss component.')
add_arg('--perceptual-weight',  default=1e0, type=float,            help='Weight for VGG-layer perceptual loss.')
add_arg('--device',             default='cpu', type=str,            help='Name of the CPU/GPU to use, for Theano.')
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
                                      'print_active_device=False'.format(args.device))

# Scientific & Imaging Libraries
import numpy as np
import scipy.ndimage, scipy.misc

# Numeric Computing (GPU)
import theano, theano.tensor as T
T.nnet.softminus = lambda x: x - T.nnet.softplus(x)

# Deep Learning Framework
import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer, Pool2DLayer as PoolLayer
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

    def setup_perceptual(self, input):
        """Use lasagne to create a network of convolution layers using pre-trained VGG19 weights.
        """
        offset = np.array([103.939, 116.779, 123.680], dtype=np.float32).reshape((1,3,1,1))
        self.network['percept'] = lasagne.layers.NonlinearityLayer(input, lambda x: ((x+0.5)*255.0) - offset)

        self.network['mse'] = self.network['percept']
        self.network['conv1_1'] = ConvLayer(self.network['percept'], 64, 3, pad=1)
        self.network['conv1_2'] = ConvLayer(self.network['conv1_1'], 64, 3, pad=1)
        self.network['pool1']   = PoolLayer(self.network['conv1_2'], 2, mode='max')
        self.network['conv2_1'] = ConvLayer(self.network['pool1'],   128, 3, pad=1)
        self.network['conv2_2'] = ConvLayer(self.network['conv2_1'], 128, 3, pad=1)
        self.network['pool2']   = PoolLayer(self.network['conv2_2'], 2, mode='max')
        self.network['conv3_1'] = ConvLayer(self.network['pool2'],   256, 3, pad=1)
        self.network['conv3_2'] = ConvLayer(self.network['conv3_1'], 256, 3, pad=1)
        self.network['conv3_3'] = ConvLayer(self.network['conv3_2'], 256, 3, pad=1)
        self.network['conv3_4'] = ConvLayer(self.network['conv3_3'], 256, 3, pad=1)
        self.network['pool3']   = PoolLayer(self.network['conv3_4'], 2, mode='max')
        self.network['conv4_1'] = ConvLayer(self.network['pool3'],   512, 3, pad=1)
        self.network['conv4_2'] = ConvLayer(self.network['conv4_1'], 512, 3, pad=1)
        self.network['conv4_3'] = ConvLayer(self.network['conv4_2'], 512, 3, pad=1)
        self.network['conv4_4'] = ConvLayer(self.network['conv4_3'], 512, 3, pad=1)
        self.network['pool4']   = PoolLayer(self.network['conv4_4'], 2, mode='max')
        self.network['conv5_1'] = ConvLayer(self.network['pool4'],   512, 3, pad=1)
        self.network['conv5_2'] = ConvLayer(self.network['conv5_1'], 512, 3, pad=1)
        self.network['conv5_3'] = ConvLayer(self.network['conv5_2'], 512, 3, pad=1)
        self.network['conv5_4'] = ConvLayer(self.network['conv5_3'], 512, 3, pad=1)


    #------------------------------------------------------------------------------------------------------------------
    # Input / Output
    #------------------------------------------------------------------------------------------------------------------

    def list_generator_layers(self):
        for l in lasagne.layers.get_all_layers(self.network['out'], treat_as_input=[self.network['img']]):
            if not l.get_params(): continue
            name = list(self.network.keys())[list(self.network.values()).index(l)]
            yield (name, l)

    def get_filename(self, absolute=False):
        filename = 'ne%ix-%s-%s-%s.pkl.bz2' % (args.zoom, args.type, args.model, __version__)
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


class NeuralEnhancer(object):

    def __init__(self):
        if len(args.files) == 0: error("Specify the image(s) to enhance on the command-line.")
        print('Enhancing {} image(s) specified on the command-line.'.format(len(args.files)))

        self.model = Model()

    def match_histograms(self, A, B, rng=(0.0, 255.0), bins=64):
        (Ha, Xa), (Hb, Xb) = [np.histogram(i, bins=bins, range=rng, density=True) for i in [A, B]]
        X = np.linspace(rng[0], rng[1], bins, endpoint=True)
        Hpa, Hpb = [np.cumsum(i) * (rng[1] - rng[0]) ** 2 / float(bins) for i in [Ha, Hb]]
        inv_Ha = scipy.interpolate.interp1d(X, Hpa, bounds_error=False, fill_value='extrapolate')
        map_Hb = scipy.interpolate.interp1d(Hpb, X, bounds_error=False, fill_value='extrapolate')
        return map_Hb(inv_Ha(A).clip(0.0, 255.0))

    def process(self, original):
        # Snap the image to a shape that's compatible with the generator (2x, 4x)
        s = 2 ** max(args.generator_upscale, args.generator_downscale)
        by, bx = original.shape[0] % s, original.shape[1] % s
        original = original[by-by//2:original.shape[0]-by//2,bx-bx//2:original.shape[1]-bx//2,:]

        # Prepare paded input image as well as output buffer of zoomed size.
        s, p, z = args.rendering_tile, args.rendering_overlap, args.zoom
        image = np.pad(original, ((p, p), (p, p), (0, 0)), mode='reflect')
        output = np.zeros((original.shape[0] * z, original.shape[1] * z, 3), dtype=np.float32)

        # Iterate through the tile coordinates and pass them through the network.
        for y, x in itertools.product(range(0, original.shape[0], s), range(0, original.shape[1], s)):
            img = np.transpose(image[y:y+p*2+s,x:x+p*2+s,:] / 255.0 - 0.5, (2, 0, 1))[np.newaxis].astype(np.float32)
            *_, repro = self.model.predict(img)
            output[y*z:(y+s)*z,x*z:(x+s)*z,:] = np.transpose(repro[0] + 0.5, (1, 2, 0))[p*z:-p*z,p*z:-p*z,:]
            print('.', end='', flush=True)
        output = output.clip(0.0, 1.0) * 255.0

        # Match color histograms if the user specified this option.
        if args.rendering_histogram:
            for i in range(3):
                output[:,:,i] = self.match_histograms(output[:,:,i], original[:,:,i])

        return scipy.misc.toimage(output, cmin=0, cmax=255)


if __name__ == "__main__":
    enhancer = NeuralEnhancer()
    for filename in args.files: # TODO: parallel processing
        print(filename, end=' ')
        img = scipy.ndimage.imread(filename, mode='RGB')
        out = enhancer.process(img)
        out.save(os.path.splitext(filename)[0]+'_ne%ix.png' % args.zoom)
        print(flush=True)
