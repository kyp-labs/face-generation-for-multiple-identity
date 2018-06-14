"""
All resources can be downloaded from:
    http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/
The implementaion is a little modified from:
    https://github.com/tkarras/progressive_growing_of_gans/blob/master/dataset_tool.py
"""

import PIL
import argparse
import numpy as np
import pandas as pd
from scipy import ndimage
from thread_pool import ThreadPool

TEST_LANDMARKS_PATH = './dataset/VGGFACE2/bb_landmark/loose_landmark_test.csv' # noqa
TEST_IMAGES_PATH = './dataset/VGGFACE2/test'
TRAIN_LANDMARKS_PATH = './dataset/VGGFACE2/bb_landmark/loose_landmark_train.csv' # noqa
TRAIN_IMAGES_PATH = './dataset/VGGFACE2/train'

# Configure all options
parser = argparse.ArgumentParser(
            description='Generate face-centered images.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add_arg = parser.add_argument
add_arg('--is_test_img', type=int, default=1,
        help='The category of the images to be processed (default: 1)')
add_arg('--start_idx', type=int, default=None,
        help='Index for the image to be firstly processed (default: None)')
add_arg('--end_idx', type=int, default=None,
        help='Index for the image to be lastly processed (default: None)')
add_arg('--resolution', type=int, default=256,
        help='Target resolution (default: 256)')
add_arg('--num_threads', type=int, default=4,
        help='Number of concurrent threads (default: 4)')
add_arg('--num_tasks', type=int, default=100,
        help='Number of concurrent processing tasks (default: 100)')

args = parser.parse_args()


# ----------------------------------------------------------------------------

def generate_face_centered_images(num_threads=4, num_tasks=100,
                                  start_idx=None, end_idx=None):
    assert(num_threads > 0 and num_tasks > 0)

    landmarks_path = TEST_LANDMARKS_PATH if args.is_test_img else TRAIN_LANDMARKS_PATH # noqa
    img_dir = TEST_IMAGES_PATH if args.is_test_img else TRAIN_IMAGES_PATH

    loose_landmarks = pd.read_csv(landmarks_path,
                                  header=None,
                                  low_memory=False)

    start_idx = 1 if start_idx is None else start_idx + 1
    end_idx = loose_landmarks[0].shape[0] - 1 if end_idx is None else end_idx + 1 # noqa

    def rot90(v):
        return np.array([-v[1], v[0]])

    def process_func(idx):
        # load original image
        img_name = loose_landmarks[0][idx]
        left_eye = loose_landmarks.T[idx][1:].values[0:2].astype('float32')
        right_eye = loose_landmarks.T[idx][1:].values[2:4].astype('float32')
        left_mouth = loose_landmarks.T[idx][1:].values[6:8].astype('float32')
        right_mouth = loose_landmarks.T[idx][1:].values[8:].astype('float32')
        img = PIL.Image.open(img_dir+'/'+img_name+'.jpg')
        print('processing ', img_name, '...')

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
        zoom = args.resolution / (np.hypot(*x) * 2)

        # Shrink.
        shrink = int(np.floor(0.5 / zoom))
        if shrink > 1:
            size = (int(np.round(float(img.size[0]) / shrink)),
                    int(np.round(float(img.size[1]) / shrink)))
            img = img.resize(size, PIL.Image.ANTIALIAS)
            quad /= shrink
            zoom *= shrink

        # Crop.
        border = max(int(np.round(args.resolution * 0.1 / zoom)), 3)
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
            pad = np.maximum(pad, int(np.round(args.resolution * 0.3 / zoom)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]),
                         (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.mgrid[:h, :w, :1]
            mask = 1.0 - np.minimum(np.minimum(np.float32(x) / pad[0],
                                    np.float32(y) / pad[1]),
                                    np.minimum(np.float32(w-1-x) / pad[2],
                                    np.float32(h-1-y) / pad[3]))
            blur = args.resolution * 0.02 / zoom
            img += (ndimage.gaussian_filter(img, [blur, blur, 0]) - img) *\
                np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) *\
                np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.round(img), 0, 255)),
                                      'RGB')
            quad += pad[0:2]

        # Transform.
        img = img.transform((args.resolution*4, args.resolution*4),
                            PIL.Image.QUAD,
                            (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        img = img.resize((args.resolution, args.resolution),
                         PIL.Image.ANTIALIAS)
        # img = np.asarray(img).transpose(2, 0, 1)

        return (img, img_name)

    with ThreadPool(args.num_threads) as pool:
        for img, img_name in pool.process_items_concurrently(
                                                range(start_idx, end_idx+1),
                                                process_func=process_func,
                                                max_items_in_flight=num_tasks):
            filename = img_dir + '/' + img_name +\
                       '_' + str(args.resolution)+'.PNG'
            img.save(filename, 'PNG')
            print('saved ', img_name, '...')


if __name__ == '__main__':
    generate_face_centered_images(args.num_threads, args.num_tasks,
                                  args.start_idx, args.end_idx)
