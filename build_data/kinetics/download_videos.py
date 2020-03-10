import numpy as np
import os
import sys
import argparse
import pdb
from tqdm import tqdm
from multiprocessing import Pool
import functools


def get_parser():
    parser = argparse.ArgumentParser(
            description='Download and save the videos from Youtube')
    parser.add_argument(
            '--csv_path', 
            default='/mnt/fs1/Dataset/kinetics/kinetics_train.csv', 
            type=str, action='store', 
            help='Path to the csv containing the information')
    parser.add_argument(
            '--save_dir', 
            default='/data5/chengxuz/Dataset/kinetics/vd_dwnld', 
            type=str, action='store', 
            help='Directory to hold the downloaded videos, rescaled')
    parser.add_argument(
            '--save_raw_dir', 
            default='/data5/chengxuz/Dataset/kinetics/vd_raw_dwnld', 
            type=str, action='store', 
            help='Directory to hold the downloaded videos, rescaled')
    parser.add_argument(
            '--min_resz', 
            default=320, type=int, action='store', 
            help='Size for the smallest edge to be resized to')
    parser.add_argument(
            '--sta_idx', default=0, type=int, action='store', 
            help='Start index for downloading')
    parser.add_argument(
            '--len_idx', default=2500, type=int, 
            action='store', help='Length of index of downloading')
    parser.add_argument(
            '--check', default=0, type=int, action='store', 
            help='Whether checking the existence')
    return parser


def load_csv(csv_path, return_cate_lbls=False):
    fin = open(csv_path, 'r')
    csv_lines = fin.readlines()
    csv_lines = csv_lines[1:]
    all_data = []

    cate_list = []

    curr_indx = 0

    for curr_line in csv_lines:
        if curr_line[-1]=='\n':
            curr_line = curr_line[:-1]
        line_split = curr_line.split(',')
        curr_cate = line_split[0]
        curr_cate = curr_cate.replace(' ', '_')
        curr_cate = curr_cate.replace('"', '')
        curr_cate = curr_cate.replace('(', '')
        curr_cate = curr_cate.replace(')', '')
        curr_cate = curr_cate.replace("'", '')
        curr_dict = {
                'cate': curr_cate, 
                'id': line_split[1], 
                'sta': int(line_split[2]), 
                'end': int(line_split[3]), 
                'train': line_split[4], 
                'flag': int(line_split[5]), 
                'indx': curr_indx}

        if not curr_dict['cate'] in cate_list:
            cate_list.append(curr_dict['cate'])

        curr_dict['cate_lbl'] = cate_list.index(curr_dict['cate'])

        all_data.append(curr_dict)
        curr_indx = curr_indx + 1
    if not return_cate_lbls:
        return all_data
    else:
        return all_data, cate_list


def make_dirs(args, cate_lbls):
    os.system('mkdir -p %s' % args.save_dir)
    os.system('mkdir -p %s' % args.save_raw_dir)
    for each_cate in cate_lbls:
        curr_save_dir = os.path.join(args.save_dir, each_cate)
        curr_save_raw_dir = os.path.join(args.save_raw_dir, each_cate)
        os.system('mkdir -p %s' % curr_save_dir)
        os.system('mkdir -p %s' % curr_save_raw_dir)


def download_one_video(curr_indx, args, csv_data):
    # Borrowed from https://github.com/nficano/pytube
    import cv2
    from pytube import YouTube
    youtube_frmt = "http://youtu.be/%s"
    curr_data = csv_data[curr_indx]
    curr_folder = curr_data['cate']
    curr_youtube_web = youtube_frmt % curr_data['id']
    try:
        yt = YouTube(curr_youtube_web)
    except:
        return 

    avi_name = '%s_%i' % (curr_data['id'], curr_data['sta'])
    avi_path = os.path.join(
            args.save_dir, curr_folder, '%s.avi' % avi_name)
    if os.path.exists(avi_path):
        if args.check==0:
            os.system('rm %s' % avi_path)
        else:
            return 

    video = yt.streams.filter(
            progressive=True, subtype='mp4', resolution='360p').first()

    mp4_folder = os.path.join(args.save_raw_dir, curr_folder)
    mp4_path = os.path.join(
            mp4_folder, '%s.mp4' % avi_name)
    if os.path.exists(mp4_path):
        os.system('rm %s' % mp4_path)
    try:
        video.download(mp4_folder)
    except:
        return
    downloaded_path = os.path.join(mp4_folder, video.default_filename)
    os.rename(downloaded_path, mp4_path)
    vidcap = cv2.VideoCapture(mp4_path)

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    start_indx = curr_data['sta'] * fps
    len_frame = (curr_data['end'] - curr_data['sta']) * fps
    vid_len = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    len_frame = int(min(vid_len - start_indx, len_frame))

    vid_height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    vid_width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)

    if vid_width < vid_height:
        vid_height = int(vid_height*1.0/vid_width*args.min_resz)
        vid_width = args.min_resz
    else:
        vid_width = int(vid_width*1.0/vid_height*args.min_resz)
        vid_height = args.min_resz

    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    vidwrt = cv2.VideoWriter(avi_path, fourcc, fps, (vid_width, vid_height))
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, int(start_indx))

    for indx_frame in range(len_frame):
        suc, frame = vidcap.read()
        if not suc:
            break
        frame_rsz = cv2.resize(frame, (vid_width, vid_height))
        vidwrt.write(frame_rsz)

    vidwrt.release()
    vidcap.release()
    os.system('rm %s' % mp4_path)


def multi_process_download(args, csv_data, curr_len):
    _func = functools.partial(download_one_video, args=args, csv_data=csv_data)
    p = Pool(30)
    r = list(tqdm(
        p.imap(
            _func, 
            range(args.sta_idx, args.sta_idx + curr_len)), 
        total=curr_len))


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    csv_data, cate_lbls = load_csv(args.csv_path, return_cate_lbls=True)
    make_dirs(args, cate_lbls)

    curr_len = min(len(csv_data) - args.sta_idx, args.len_idx)
    #for curr_indx in tqdm(range(args.sta_idx, args.sta_idx + curr_len)):
    #    download_one_video(curr_indx, args, csv_data)
    multi_process_download(args, csv_data, curr_len)


if __name__ == '__main__':
    main()
