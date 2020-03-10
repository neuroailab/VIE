import numpy as np
import os
import sys
import argparse
import pdb
from tqdm import tqdm
from multiprocessing import Pool
import functools

from download_videos import load_csv
import shutil


def get_parser():
    parser = argparse.ArgumentParser(
            description='The script to build meta data from jpgs, '\
                    'also removing empty folders')
    parser.add_argument(
            '--csv_path', 
            default='/mnt/fs1/Dataset/kinetics/kinetics_train.csv', 
            type=str, action='store', 
            help='Path to the csv containing the information')
    parser.add_argument(
            '--jpg_dir', 
            default='/data5/chengxuz/Dataset/kinetics/jpgs_extracted',
            type=str, action='store', 
            help='Directory to hold the extracted jpgs, rescaled')
    return parser


def check_folder_name(curr_indx, args, csv_data):
    curr_data = csv_data[curr_indx]
    folder_name = '%s_%i' %  (curr_data['id'], curr_data['sta'])
    jpg_folder = os.path.join(
            args.jpg_dir, 
            curr_data['cate'], folder_name)
    if not os.path.exists(jpg_folder) and os.path.exists(jpg_folder[:-1]):
        shutil.move(jpg_folder[:-1], jpg_folder)
    if not os.path.exists(jpg_folder) and os.path.exists(jpg_folder[:-2]):
        shutil.move(jpg_folder[:-2], jpg_folder)
    if not os.path.exists(jpg_folder) and os.path.exists(jpg_folder[:-3]):
        shutil.move(jpg_folder[:-3], jpg_folder)

def main():
    parser = get_parser()
    args = parser.parse_args()

    csv_data = load_csv(args.csv_path)
    data_len = len(csv_data)

    for idx in tqdm(range(data_len)):
        check_folder_name(idx, args, csv_data)

if __name__ == '__main__':
    main()
