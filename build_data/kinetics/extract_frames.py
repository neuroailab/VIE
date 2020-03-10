import numpy as np
import math
import os
import sys
import argparse
import pdb
from tqdm import tqdm
from multiprocessing import Pool
import functools
import cv2

from download_videos import load_csv

NUM_THREADS = 10


def get_parser():
    parser = argparse.ArgumentParser(
            description='The script to extract the jpgs from videos')
    parser.add_argument(
            '--csv_path',
            default='/mnt/fs1/Dataset/kinetics/kinetics_train.csv',
            type=str, action='store',
            help='Path to the csv containing the information')
    parser.add_argument(
            '--video_dir',
            default='/data5/chengxuz/Dataset/kinetics/vd_dwnld',
            type=str, action='store',
            help='Directory to hold the downloaded videos')
    parser.add_argument(
            '--jpg_dir',
            default='/data5/chengxuz/Dataset/kinetics/jpgs_extracted_test',
            type=str, action='store',
            help='Directory to hold the extracted jpgs, rescaled')
    parser.add_argument(
            '--sta_idx', default=0, type=int, action='store',
            help='Start index for downloading')
    parser.add_argument(
            '--len_idx', default=100, type=int,
            action='store', help='Length of index of downloading')
    parser.add_argument(
            '--check', default=0, type=int, action='store',
            help='Whether checking the existence')
    parser.add_argument(
            '--remove_empty', action='store_true',
            help='Whether just remove empty folders')
    return parser


def extract_one_video(curr_indx, args, csv_data):
    curr_data = csv_data[curr_indx]
    avi_name = '%s_%i.avi' % (curr_data['id'], curr_data['sta'])
    save_folder = os.path.join(
            args.jpg_dir,
            curr_data['cate'], avi_name.rstrip('.avi'))
    avi_path = os.path.join(args.video_dir, curr_data['cate'], avi_name)
    if args.remove_empty:
        if os.path.exists(save_folder) and not os.path.exists(avi_path):
            os.rmdir(save_folder)
        return

    if os.path.exists(save_folder) and args.check==1:
        return

    if not os.path.exists(avi_path):
        return
    os.system('mkdir -p %s' % save_folder)

    vidcap = cv2.VideoCapture(avi_path)
    vid_height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    vid_width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)

    if vid_width < vid_height:
        resolution_str = '320:-1'
    else:
        resolution_str = '-1:320'

    tmpl = '%06d.jpg'
    cmd = 'ffmpeg -i {} -vf scale={},fps=25 {} > /dev/null 2>&1'.format(
            avi_path,
            resolution_str,
            os.path.join(save_folder, tmpl))
    os.system(cmd)


def main():
    parser = get_parser()
    args = parser.parse_args()

    csv_data = load_csv(args.csv_path)
    curr_len = min(len(csv_data) - args.sta_idx, args.len_idx)

    _func = functools.partial(extract_one_video, args=args, csv_data=csv_data)
    p = Pool(NUM_THREADS)
    r = list(tqdm(
        p.imap(
            _func,
            range(args.sta_idx, args.sta_idx + curr_len)),
        total=curr_len))


if __name__ == '__main__':
    main()
