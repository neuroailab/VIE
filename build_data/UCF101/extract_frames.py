import argparse
import os
import sys
from tqdm import tqdm
from multiprocessing import Pool
import functools
import cv2

NUM_THREADS = 10


def get_parser():
    parser = argparse.ArgumentParser(
            description='The script to extract the jpgs from videos')
    parser.add_argument(
            '--raw_video_dir',
            type=str, action='store',
            help='Directory to hold the downloaded videos')
    parser.add_argument(
            '--jpg_dir',
            type=str, action='store',
            help='Directory to hold the extracted jpgs, rescaled')
    parser.add_argument(
            '--check', default=0, type=int, action='store',
            help='Whether checking the existence')
    parser.add_argument(
            '--remove_empty', action='store_true',
            help='Whether just remove empty folders')
    return parser


def extract_one_video(video_name, args):
    if not video_name.endswith(".avi"):
        return
    
    save_path = os.path.join(
            args.jpg_dir, video_name[:-4])
    video_path = os.path.join(args.raw_video_dir, video_name)
    
    if args.remove_empty:
        if os.path.exists(save_path) and not os.path.exists(video_path):
            os.rmdir(save_path)
        return
    if os.path.exists(save_path) and args.check==1:
        return
    if not os.path.exists(video_path):
        return
    
    os.system('mkdir -p %s' % save_path)

    vidcap = cv2.VideoCapture(video_path)
    vid_height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    vid_width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)

    if vid_width < vid_height:
        resolution_str = '320:-1'
    else:
        resolution_str = '-1:320'

    tmpl = '%06d.jpg'
    cmd = 'ffmpeg -i {} -vf scale={},fps=25 {} > /dev/null 2>&1'.format(
            video_path,
            resolution_str,
            os.path.join(save_path, tmpl))
    os.system(cmd)


def main():
    parser = get_parser()
    args = parser.parse_args()
 
    video_names = os.listdir(args.raw_video_dir)

    _func = functools.partial(extract_one_video, args=args)
    p = Pool(NUM_THREADS)
    r = list(tqdm(
        p.imap(_func, video_names),
        total=len(video_names)))  


if __name__ == '__main__':
    main()
