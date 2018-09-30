import os
import glob
import argparse

import pandas as pd


class CsvMerger(object):
    """Merge csv"""

    def __init__(self,
                 data_dir,
                 landmark_save_path,
                 filtered_list_save_path):
        self.landmark_save_path = landmark_save_path
        self.filtered_list_save_path = filtered_list_save_path
        self.people_dirs = glob.glob(data_dir + '/n[0-9]*')

    def save_merged_landmark(self):
        all_landmarks = pd.DataFrame([])
        for d in self.people_dirs:
            landmark = pd.read_csv(os.path.join(d, "loose_landmarks_256.csv"))
            all_landmarks = all_landmarks.append(landmark)
        all_landmarks.to_csv(self.landmark_save_path, index=False)
        return

    def save_merged_filtered_list(self):
        all_filtered_list = pd.DataFrame([])
        for d in self.people_dirs:
            filtered_list = pd.read_excel(os.path.join(d, "result.xlsx"))
            all_filtered_list = all_filtered_list.append(filtered_list)
        all_filtered_list.to_csv(self.filtered_list_save_path, index=False)
        return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default="../dataset/VGGFACE2/train/raw",
                        help="Directory containing images", type=str)
    parser.add_argument("--landmark_save_path",
                        default="../dataset/VGGFACE2/train/all_loose_landmarks_256.csv",
                        help="All merged loose landmarks", type=str)
    parser.add_argument("--filtered_list_save_path",
                        default="../dataset/VGGFACE2/train/all_filtered_results.csv",
                        help="Filtered results", type=str)
    args = parser.parse_args()

    merger = CsvMerger(args.data_dir,
                       args.landmark_save_path,
                       args.filtered_list_save_path)

    merger.save_merged_landmark()
    merger.save_merged_filtered_list()
