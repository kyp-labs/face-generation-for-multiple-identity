"""
All resources can be downloaded from:
    http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/
The implementaion is a little modified from:
    https://github.com/tkarras/progressive_growing_of_gans/blob/master/dataset_tool.py
"""
# flake8: noqa

import os
import PIL
import argparse
import numpy as np
import pandas as pd
from scipy import ndimage
from thread_pool import ThreadPool

TEST_LANDMARKS_PATH = './dataset/VGGFACE2/bb_landmark/loose_landmark_test.csv' # noqa
TRAIN_LANDMARKS_PATH = './dataset/VGGFACE2/bb_landmark/loose_landmark_train.csv' # noqa
IDENTITY_INFO_PATH = './identity_info.csv'
IMAGES_PATH = './dataset/hr_images'
OUTPUT_PATH = './output'

# Configure all options
parser = argparse.ArgumentParser(
            description='Generate face-centered images.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add_arg = parser.add_argument
add_arg('--identity-info', type=str, default=IDENTITY_INFO_PATH,
        help='Path of identity_info.csv')
add_arg('--resolution', type=int, default=256,
        help='Target resolution (default: 256)')
add_arg('--scale', type=int, default=4,
        help='Scale by super-resolution (default: 4)')
add_arg('--num-threads', type=int, default=4,
        help='Number of concurrent threads (default: 4)')
add_arg('--num-tasks', type=int, default=100,
        help='Number of concurrent processing tasks (default: 100)')

args = parser.parse_args()


# ----------------------------------------------------------------------------

def generate_face_centered_images(loose_landmarks, img_dir, outdir,
                                  num_threads=4, num_tasks=100, scale=4,
                                  start_idx=None, end_idx=None):
    assert(num_threads > 0 and num_tasks > 0)

    def rot90(v):
        return np.array([-v[1], v[0]])

    def resize_landmarks(landmarks, quad):
        size = ((quad[3][0] + quad[2][0]) - (quad[0][0] + quad[1][0])) / 2
        return landmarks * args.resolution / size

    def rotate_landmarks(landmarks, quad):
        hypot_vec = quad[1] - quad[0]
        bottom_vec = np.array([0., hypot_vec[1]])
        height_vec = hypot_vec - bottom_vec
        sin = np.hypot(*height_vec) / np.hypot(*hypot_vec)
        cos = np.hypot(*bottom_vec) / np.hypot(*hypot_vec)

        # Rotate clockwise
        if height_vec[0] > 0:
            sin *= -1.

        rotation_matrix = np.matrix([[cos, -sin], [sin, cos]])

        return landmarks.dot(rotation_matrix)

    def process_func(idx):
        # load original image
        img_name = loose_landmarks[0][idx]
        class_id, img_name = img_name.split('/')

        left_eye = loose_landmarks.T[idx][1:].values[0:2].astype('float32') * scale
        right_eye = loose_landmarks.T[idx][1:].values[2:4].astype('float32') * scale
        nose = loose_landmarks.T[idx][1:].values[4:6].astype('float32') * scale
        left_mouth = loose_landmarks.T[idx][1:].values[6:8].astype('float32') * scale
        right_mouth = loose_landmarks.T[idx][1:].values[8:].astype('float32') * scale
        landmarks = np.stack([left_eye, right_eye, nose, left_mouth, right_mouth])
        try:
            img = PIL.Image.open(img_dir+'/'+class_id+'-'+img_name+'.png')
        except FileNotFoundError:
            print(img_dir+'/'+class_id+'-'+img_name+'.png does not exist')
            return -1
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
            landmarks /= shrink
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
            landmarks -= crop[0:2]

        # Simulate super-resolution.
        superres = int(np.exp2(np.ceil(np.log2(zoom))))
        if superres > 1:
            img = img.resize((img.size[0] * superres, img.size[1] * superres),
                             PIL.Image.ANTIALIAS)
            quad *= superres
            landmarks *= superres
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
            landmarks += pad[0:2]

        # Transform.
        quad += 0.5
        img = img.transform((args.resolution*4, args.resolution*4),
                            PIL.Image.QUAD,
                            quad.flatten(), PIL.Image.BILINEAR)
        img = img.resize((args.resolution, args.resolution),
                         PIL.Image.ANTIALIAS)


        landmarks -= quad[0]
        landmarks = rotate_landmarks(landmarks, quad)
        landmarks = resize_landmarks(landmarks, quad)

        # img = np.asarray(img).transpose(2, 0, 1)

        # Save new landmarks
        print(img_name, landmarks)
        for i in range(5):
            loose_landmarks.T[idx][1:].values[i*2:i*2+2] = landmarks[i]

        return (img, img_name)

    with ThreadPool(args.num_threads) as pool:
        for ret in pool.process_items_concurrently(
                                                range(start_idx, end_idx+1),
                                                process_func=process_func,
                                                max_items_in_flight=num_tasks):
            if ret != -1:
                img, img_name = ret
                filename = outdir + '/' + img_name + '.png'
                img.save(filename, 'PNG')
                print('saved ', img_name, '...')


if __name__ == '__main__':
    assert(os.path.isdir(IMAGES_PATH))
    assert(os.path.exists(args.identity_info))
    identity_info = pd.read_csv(args.identity_info)

    assert(os.path.exists(TEST_LANDMARKS_PATH))
    assert(os.path.exists(TRAIN_LANDMARKS_PATH))
    print('loading landmarks.csv...')
    loose_landmarks_test = pd.read_csv(TEST_LANDMARKS_PATH,
                                       header=None,
                                       low_memory=False)
    loose_landmarks_train = pd.read_csv(TRAIN_LANDMARKS_PATH,
                                        header=None,
                                        low_memory=False)

    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    if not os.path.exists(OUTPUT_PATH+'/'+str(args.resolution)):
        os.mkdir(OUTPUT_PATH+'/'+str(args.resolution))

    for i in range(len(identity_info['Class_ID'])):
        class_id = identity_info['Class_ID'][i]
        name = identity_info['Name'][i]
        print('processing identity: ', class_id, name, '...')

        out_dir = OUTPUT_PATH + '/' + str(args.resolution) + '/' + class_id
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        is_train = identity_info['Flag'][i]
        start_idx = identity_info['Start_Idx'][i] + 1
        end_idx = start_idx + identity_info['Sample_Num'][i] - 1

        loose_landmarks = loose_landmarks_train if is_train else loose_landmarks_test

        generate_face_centered_images(loose_landmarks, IMAGES_PATH, out_dir,
                                      args.num_threads, args.num_tasks, args.scale,
                                      start_idx, end_idx)

    print('saving landmarks.csv...')
    loose_landmarks_train.to_csv(OUTPUT_PATH + '/' + 'loose_landmarks_train_'
                                         + str(args.resolution) + '.csv')
    loose_landmarks_test.to_csv(OUTPUT_PATH + '/' + 'loose_landmarks_test_'
                                        + str(args.resolution) + '.csv')
