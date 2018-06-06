# All resources can be downloaded from:
#   http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/
# The implementaion is a little modified from:
#   https://github.com/tkarras/progressive_growing_of_gans/blob/master/dataset_tool.py

import os
import PIL
import argparse
import pandas as pd
import numpy as np
from scipy import ndimage

# TODO: Remove later
PATH_LANDMARKS = './dataset/VGGFACE2/bb_landmark/loose_landmark_test.csv'
PATH_IMAGES = './dataset/VGGFACE2/test'

# Configure all options
parser = argparse.ArgumentParser(
            description='Generate face-centered images.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add_arg = parser.add_argument
add_arg('--landmarks', type=str, default=PATH_LANDMARKS,  # TODO: required=True
        help='Path of csv file for landmarks')
add_arg('--image_dir', type=str, default=PATH_IMAGES,  # TODO: required=True
        help='Path of directory for images')
add_arg('--resolution', type=int, default=256,
        help='Target resolution')
add_arg('--idx', type=int, default=256,  # TODO: Remove after testing
        help='idx for testing')
args = parser.parse_args()


def rot90(v):
    return np.array([-v[1], v[0]])


def process(idx, loose_landmarks, image_dir, resolution):
    # load original image
    img_name = loose_landmarks[0][idx]
    left_eye = loose_landmarks.T[idx][1:].values[0:2].astype('float32')
    right_eye = loose_landmarks.T[idx][1:].values[2:4].astype('float32')
    left_mouth = loose_landmarks.T[idx][1:].values[6:8].astype('float32')
    right_mouth = loose_landmarks.T[idx][1:].values[8:].astype('float32')
    img = PIL.Image.open(image_dir+'/'+img_name+'.jpg')

    # Choose oriented crop rectangle.
    eye_avg = (left_eye + right_eye) * 0.5 + 0.5
    mouth_avg = (left_mouth + right_mouth) * 0.5 + 0.5
    eye_to_eye = right_mouth - left_mouth
    eye_to_mouth = mouth_avg - eye_avg
    x = eye_to_eye - rot90(eye_to_mouth)
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = rot90(x)
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    zoom = resolution / (np.hypot(*x) * 2)

    # Shrink.
    shrink = int(np.floor(0.5 / zoom))
    if shrink > 1:
        size = (int(np.round(float(img.size[0]) / shrink)),
                int(np.round(float(img.size[1]) / shrink)))
        img = img.resize(size, PIL.Image.ANTIALIAS)
        quad /= shrink
        zoom *= shrink

    # Crop.
    border = max(int(np.round(resolution * 0.1 / zoom)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0),
            max(crop[1] - border, 0),
            min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Simulate super-resolution.
    superres = int(np.exp2(np.ceil(np.log2(zoom))))
    if superres > 1:
        img = img.resize((img.size[0] * superres, img.size[1] * superres),
                         PIL.Image.ANTIALIAS)
        quad *= superres
        zoom /= superres

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
           int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0),
           max(-pad[1] + border, 0),
           max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if max(pad) > border - 4:
        pad = np.maximum(pad, int(np.round(resolution * 0.3 / zoom)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]),
                     (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.mgrid[:h, :w, :1]
        mask = 1.0 - np.minimum(np.minimum(np.float32(x) / pad[0],
                                np.float32(y) / pad[1]),
                                np.minimum(np.float32(w-1-x) / pad[2],
                                np.float32(h-1-y) / pad[3]))
        blur = resolution * 0.02 / zoom
        img += (ndimage.gaussian_filter(img, [blur, blur, 0]) - img) *\
            np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.round(img), 0, 255)),
                                  'RGB')
        quad += pad[0:2]

    # Transform.
    img = img.transform((resolution*4, resolution*4), PIL.Image.QUAD,
                        (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    img = img.resize((resolution, resolution), PIL.Image.ANTIALIAS)
    # img = np.asarray(img).transpose(2, 0, 1)

    return img


if __name__ == '__main__':
    assert(os.path.exists(args.landmarks))
    assert(os.path.isdir(args.image_dir))

    loose_landmarks = pd.read_csv(args.landmarks,
                                  header=None,
                                  low_memory=False)
    img_name = loose_landmarks[0][args.idx]  # TODO: Remove later
    print('Target image: ', img_name)

    orig_img = PIL.Image.open(args.image_dir+'/'+img_name+'.jpg')
    proc_img = process(args.idx, loose_landmarks,
                       args.image_dir, args.resolution)

    orig_img.save('./'+str(args.idx)+'_orig.JPEG', 'JPEG')
    proc_img.save('./'+str(args.idx)+'_proc.JPEG', 'JPEG')
