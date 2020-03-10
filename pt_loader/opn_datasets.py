import os
import os.path
from collections import namedtuple
import time
import pdb

import numpy as np
import torch.utils.data as data
from PIL import Image

from pt_loader.datasets import VideoRecord


class OPNVideoDataset(data.Dataset):
    MIN_NUM_FRAMES = 16
    T_MAX_CHOICES = [9, 15]

    def __init__(self, root, metafile, file_tmpl='{:06d}.jpg', transform=None):
        self.root = root
        self.metafile = metafile
        self.transform = transform
        self.file_tmpl = file_tmpl

        self._parse_list()

    def _parse_list(self):
        # check the frame number is >= MIN_NUM_FRAMES
        # usualy it is [video_id, num_frames, class_idx]
        with open(self.metafile) as f:
            lines = [x.strip().split(' ') for x in f]
            lines = [line for line in lines
                     if int(line[1]) >= self.MIN_NUM_FRAMES]

        self.video_list = [VideoRecord(*v) for v in lines]
        print('Number of videos: {}'.format(len(self.video_list)))

    def _get_valid_video(self, index):
        record = self.video_list[index]
        # check this is a legit video folder
        while not os.path.exists(
                os.path.join(self.root, record.path, self.file_tmpl.format(1))):
            print(
                os.path.join(
                    self.root,
                    record.path,
                    self.file_tmpl.format(1)))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
        return record, index

    def _load_image(self, directory, idx):
        tmpl = os.path.join(self.root, directory, self.file_tmpl)
        try:
            return Image.open(tmpl.format(idx)).convert('RGB')
        except Exception:
            print('error loading image: {}'.format(tmpl.format(idx)))
            return Image.open(tmpl.format(1)).convert('RGB')

    def _get_indices(self, record):
        rec_no_frames = int(record.num_frames)
        t_max = np.random.choice(self.T_MAX_CHOICES)
        start_idx = np.random.randint(rec_no_frames - t_max)
        indices = [start_idx + _tmp_idx * t_max // 3 for _tmp_idx in range(4)]
        return np.asarray(indices) + 1

    def __getitem__(self, index):
        record, index = self._get_valid_video(index)
        indices = self._get_indices(record)
        frames = self.transform([self._load_image(record.path, int(idx))
                                 for idx in indices])
        return frames

    def __len__(self):
        return len(self.video_list)


def normalize(x):
    x -= x.min()
    #m = x.max()
    #x /= m if m != 0 else 1
    if x.max() == 0:
        x += 1
    x /= x.sum()
    return x


class MotionAwareOPNVideoDataset(OPNVideoDataset):

    magnitude_templ = 'magnitudes.npy'

    def __init__(self, root, flow_root, metafile, file_tmpl='{:06d}.jpg',
                 transform=None):
        self.root = root
        self.flow_root = flow_root
        self.metafile = metafile
        self.transform = transform
        self.file_tmpl = file_tmpl

        self._parse_list()

    def _load_magnitudes(self, record):
        mag_path = os.path.join(self.flow_root,
                                record.path,
                                self.magnitude_templ)
        try:
            mag_arr = np.load(mag_path)
        except:
            print(mag_path, "Mag not there!")
            mag_arr = np.ones(int(record.num_frames))

        if len(mag_arr) == 0:
            mag_arr = np.ones(int(record.num_frames))
        return mag_arr

    def _get_indices(self, record):
        t_max = np.random.choice(self.T_MAX_CHOICES)
        magnitudes = self._load_magnitudes(record)
        window_weights = np.convolve(magnitudes, np.ones(t_max), mode='valid')
        window_weights = normalize(window_weights)

        rec_no_frames = int(record.num_frames)
        start_idx = np.random.choice(
                len(window_weights),
                p=window_weights)
        start_idx = min(start_idx, rec_no_frames - t_max - 1)
        indices = [start_idx + _tmp_idx * t_max // 3 for _tmp_idx in range(4)]
        return np.asarray(indices) + 1


if __name__ == '__main__':
    import config
    import transforms
    import torch

    root = '/mnt/fs3/chengxuz/kinetics/pt_meta'
    root_data = '/data5/chengxuz/Dataset/kinetics/comp_jpgs_extracted'
    cfg = config.dataset_config('kinetics', root=root, root_data=root_data)
    transform = transforms.video_OPN_transform_color()
    dataset = OPNVideoDataset(
        cfg['root'], cfg['train_metafile'], transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True,
        num_workers=10, pin_memory=False,
        worker_init_fn=lambda x: np.random.seed(x))

    curr_time = time.time()
    init_time = curr_time
    data_enumerator = enumerate(dataloader)
    for i in range(100):
        _, input = data_enumerator.next()
        print(input.shape, input.dtype, np.max(input.numpy()))
        curr_time = time.time()
    print(time.time() - init_time)
