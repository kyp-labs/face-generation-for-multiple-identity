"""Image generator.

Result directory structure:
    ./datasets
        - VGGFACE2
            - train
                - raw
                    - n000810
                    - n000001
                    - ...
                - 4
                - 8
                - ...
                - all_filtered_results.csv
                - all_loose_landmarks_256.csv

python image_generator.py
"""

import os
import argparse

import glob
import cv2


class ResizedImageSaver(object):
    """Resized image saver."""

    def __init__(self,
                 data_dir,
                 save_dir,
                 resolutions_to=(16, 32),
                 img_format='png'):
        """constructor.

        Args:
            data_dir (str): Directory path containing datasets.
            save_dir (str): Directory path saving resized datasets.
            resolutions_to (list): Output image resolutions list.
            img_format (str): 'jpg' or 'png'
            person_name (str): Specific name want to use for filename,
                               Default is None.
        """
        for res in resolutions_to:
            dir_list = glob.glob(data_dir + '/n[0-9]*')
            for d in dir_list:
                file_list = glob.glob(d + f'/*.{img_format}')
                imgs = [cv2.imread(i) for i in file_list]
                imgs_resized = self.resize_image(imgs, res)
                cls_id = os.path.basename(d)
                save_cls_dir = os.path.join(save_dir, str(res), cls_id)

                if not os.path.exists(save_cls_dir):
                    os.makedirs(save_cls_dir)

                self.save_images(file_list, save_cls_dir, imgs_resized)

    def resize_image(self, imgs, res_to):
        """Image resize to target resolution.

        Args:
            resolutions_to (int): target resolutions.

        Return: resized images of target resolution.
        """
        return [cv2.resize(i, dsize=(res_to, res_to))
                for i in imgs]

    def save_images(self, file_list, save_dir, imgs_resized):
        """Save images."""
        for f, img in zip(file_list, imgs_resized):
            file_name = os.path.basename(f)
            save_path = os.path.join(save_dir, file_name)
            cv2.imwrite(save_path, img)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default="../dataset/VGGFACE2/train/raw",
                        help="Directory containing images", type=str)
    parser.add_argument("--save_dir",
                        default="../dataset/VGGFACE2/train",
                        help="Directory saving resized images", type=str)
    parser.add_argument("--resolutions_to",
                        default=[4, 8, 16, 32, 64, 128, 256],
                        help="resolutions want to resize", type=list)
    args = parser.parse_args()

    ResizedImageSaver(data_dir=args.data_dir,
                      save_dir=args.save_dir,
                      resolutions_to=args.resolutions_to)
