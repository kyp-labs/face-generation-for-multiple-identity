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
TEST_IMAGES_PATH = './dataset/VGGFACE2/test'
TRAIN_IMAGES_PATH = './dataset/VGGFACE2/train'
IDENTITY_INFO_PATH = './identity_info.csv'
IMAGES_PATH = '/media/lab4all/de27e1a3-1c5e-44b1-b884-b84f6bc55204/hr_output/dcscn_L12_F196to48_Sc3_NIN_A64_PS_R1F32'
OUTPUT_PATH = '/media/lab4all/de27e1a3-1c5e-44b1-b884-b84f6bc55204/face-centered-images'
SIZE_CHECK_IMAGE_PATH = '/media/lab4all/de27e1a3-1c5e-44b1-b884-b84f6bc55204/deblocked_output'

# Configure all options
parser = argparse.ArgumentParser(
            description='Generate face-centered images.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add_arg = parser.add_argument
add_arg('--identity-info', type=str, default=IDENTITY_INFO_PATH,
        help='Path of identity_info.csv')
add_arg('--image-path', type=str, default=IMAGES_PATH,
        help='Source image path')
add_arg('--output-path', type=str, default=OUTPUT_PATH,
        help='Output image path')
add_arg('--size_check_img_dir', type=str, default=SIZE_CHECK_IMAGE_PATH,
        help='Image path for size check')
add_arg('--resolution', type=int, default=256,
        help='Target resolution (default: 256)')
add_arg('--scale', type=int, default=3,
        help='Scale by super-resolution (default: 3)')
add_arg('--num-threads', type=int, default=4,
        help='Number of concurrent threads (default: 4)')
add_arg('--num-tasks', type=int, default=100,
        help='Number of concurrent processing tasks (default: 100)')

args = parser.parse_args()


# ----------------------------------------------------------------------------

def generate_face_centered_images(loose_landmarks, img_dir, outdir,
                                  num_threads=4, num_tasks=100, scale=4,
                                  size_check_img_dir=None, start_idx=None, end_idx=None):
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
        img_name = loose_landmarks[idx:idx+1]['NAME_ID'].values[0]

        try:
            img_name_for_size_check = img_name.replace('/', '-')
            img = PIL.Image.open(size_check_img_dir+'/'+img_name_for_size_check+'.png')
        except FileNotFoundError:
            print(size_check_img_dir+'/'+img_name_for_size_check+'.png does not exist')
            return -1

        denominator = 1
        if img.size[0] + img.size[1] >= 1024:
            denominator = scale

        class_id, img_name = img_name.split('/')
        if os.path.exists(outdir + '/' + img_name + '.png'):
            print("File already exists in the target directory")
            return -1

        try:
            img = PIL.Image.open(img_dir+'/'+class_id+'-'+img_name+'.png')
        except FileNotFoundError:
            print(img_dir+'/'+class_id+'-'+img_name+'.png does not exist')
            return -1
        print('processing ', img_name, '...')

        left_eye = loose_landmarks[idx:idx+1].values[0][1:3].astype('float32') * scale / denominator
        right_eye = loose_landmarks[idx:idx+1].values[0][3:5].astype('float32') * scale / denominator
        nose = loose_landmarks[idx:idx+1].values[0][5:7].astype('float32') * scale / denominator
        left_mouth = loose_landmarks[idx:idx+1].values[0][7:9].astype('float32') * scale / denominator
        right_mouth = loose_landmarks[idx:idx+1].values[0][9:11].astype('float32') * scale / denominator
        landmarks = np.stack([left_eye, right_eye, nose, left_mouth, right_mouth])

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
        key_values = dict({'P1X': landmarks[0,0], 'P1Y': landmarks[0,1],
                          'P2X': landmarks[1,0], 'P2Y': landmarks[1,1],
                          'P3X': landmarks[2,0], 'P3Y': landmarks[2,1],
                          'P4X': landmarks[3,0], 'P4Y': landmarks[3,1],
                          'P5X': landmarks[4,0], 'P5Y': landmarks[4,1]})
        for key, value in key_values.items():
            loose_landmarks[idx:idx+1][key] = value

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

#    for idx in range(start_idx, end_idx+1):
#        ret = process_func(idx)
#        if ret != -1:
#            img, img_name = ret
#            filename = outdir + '/' + img_name + '.png'
#            img.save(filename, 'PNG')
#            print('saved ', img_name, '...')


if __name__ == '__main__':
    assert(os.path.isdir(args.image_path))
    assert(os.path.exists(args.identity_info))
    identity_info = pd.read_csv(args.identity_info)

    assert(os.path.exists(TEST_LANDMARKS_PATH))
    assert(os.path.exists(TRAIN_LANDMARKS_PATH))
    print('loading landmarks.csv...')
    loose_landmarks_test = pd.read_csv(TEST_LANDMARKS_PATH)
    loose_landmarks_train = pd.read_csv(TRAIN_LANDMARKS_PATH)

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    if not os.path.exists(args.output_path+'/'+str(args.resolution)):
        os.mkdir(args.output_path+'/'+str(args.resolution))

    for i in range(len(identity_info['Class_ID'])):
        class_id = identity_info['Class_ID'][i]
        name = identity_info['Name'][i]
        print('processing identity: ', class_id, name, '...')

        out_dir = args.output_path + '/' + str(args.resolution) + '/' + class_id
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        is_train = identity_info['Flag'][i]
        start_idx = identity_info['Start_Idx'][i]
        end_idx = start_idx + identity_info['Sample_Num'][i] - 1

        loose_landmarks = loose_landmarks_train if is_train else loose_landmarks_test

        generate_face_centered_images(loose_landmarks, args.image_path, out_dir,
                                      args.num_threads, args.num_tasks, args.scale,
                                      args.size_check_img_dir, start_idx, end_idx)

        print('saving landmarks.csv...')
        csv_path = out_dir + '/loose_landmarks_' + str(args.resolution) + '.csv'
        if os.path.exists(csv_path):
            print("CSV File already exists in the target directory")
        else:
            loose_landmarks[start_idx:end_idx+1].to_csv(csv_path)
