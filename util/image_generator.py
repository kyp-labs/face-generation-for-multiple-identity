"""Image generator."""

import os
import argparse

import glob
import cv2


class ResizedImageSaver(object):
    """Resized image saver."""

    def __init__(self, data_dir, modify_filename=True,
                 resolutions_to=(16, 32), img_format='jpg',
                 person_name=None):
        """constructor.

        Args:
            data_dir (str): Directory path containing dataset.
            modify_filename (str): Change filename according to
                                   person's name or not.
            resolutions_to (list): Output image resolutions list.
            img_format (str): 'jpg' or 'png'
            person_name (str): Specific name want to use for filename,
                               Default is None.
        """
        self.file_list = glob.glob(data_dir + f'/*.{img_format}')
        self.num_images = len(self.file_list)
        self.images = [cv2.imread(i) for i in self.file_list]
        self.img_format = img_format

        if person_name is None:
            self.person_name = os.path.basename(data_dir)

        if img_format is None:
            self.img_format = self.file_list[0].split('.')[-1]

        assert isinstance(resolutions_to, list), \
            "resolutions_to should be list"
        for res in resolutions_to:
            self.images_resized = self.resize_image(res)

            if modify_filename:
                self.file_names = self.modify_filename()

            self.save_dir = os.path.join(data_dir, str(res))
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            self.save_images()

    def modify_filename(self):
        """Filename modifier."""
        image_names = [f'{self.person_name}_{i}.{self.img_format}'
                       for i in range(self.num_images)]
        return image_names

    def resize_image(self, resolutions_to):
        """Image resize to target resolution.

        Args:
            resolutions_to (int): target resolutions.

        Return: resized images of target resolution.
        """
        return [cv2.resize(i, dsize=(resolutions_to, resolutions_to))
                for i in self.images]

    def save_images(self):
        """Save images."""
        for file_name, image in zip(self.file_names, self.images_resized):
            save_path = os.path.join(self.save_dir, file_name)
            cv2.imwrite(save_path, image)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../test_data/aeron_judge",
                        help="Directory containing images", type=str)
    parser.add_argument("--resolutions_to",
                        default=[4, 8, 16, 32, 64, 128, 256, 512],
                        help="resolutions want to resize", type=list)
    args = parser.parse_args()

    ResizedImageSaver(data_dir=args.data_dir,
                      resolutions_to=args.resolutions_to)
