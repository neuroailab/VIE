import os
import os.path
from collections import namedtuple
import time
import pdb
import random

import numpy as np
import torch.utils.data as data
from PIL import Image

VideoRecord = namedtuple('VideoRecord',
                         ['path', 'num_frames', 'label'])

VideoCollageRecord = namedtuple('VideoCollageRecord',
                                ['path', 'num_frames', 'size', 'label'])
# `size` is tuple: (height, width)


class VideoDataset(data.Dataset):
    '''
    Build pytorch data provider for loading frames from videos

    Args:
        root (str):
            Path to the folder including all jpgs
        metafile (str):
            Path to the metafiles
        frame_interval (int):
            number of frames to skip between two sampled frames, None means
            interval will be computed so that the frames subsampled cover the
            whole video
        frame_start (str):
            Methods of determining which frame to start, RANDOM means randomly
            choosing the starting index, None means the middle of valid range
    '''

    MIN_NUM_FRAMES = 3

    def __init__(
            self, root, metafile,
            num_frames=8, frame_interval=None, frame_start=None,
            file_tmpl='{:06d}.jpg', transform=None, sample_groups=1,
            bin_interval=None,
            trn_style=False, trn_num_frames=8,
            part_vd=None, HMDB_sample=False, resnet3d_test_sample=False):

        self.root = root
        self.metafile = metafile
        self.file_tmpl = file_tmpl
        self.transform = transform
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.frame_start = frame_start
        self.sample_groups = sample_groups
        self.bin_interval = bin_interval
        self.trn_style = trn_style
        self.trn_num_frames = trn_num_frames
        self.part_vd = part_vd
        self.HMDB_sample = HMDB_sample
        self.resnet3d_test_sample = resnet3d_test_sample

        self._parse_list()

    def _load_image(self, directory, idx):
        tmpl = os.path.join(self.root, directory, self.file_tmpl)
        try:
            return Image.open(tmpl.format(idx)).convert('RGB')
        except Exception:
            print('error loading image: {}'.format(tmpl.format(idx)))
            return Image.open(tmpl.format(1)).convert('RGB')

    def __get_interval_valid_range(self, rec_no_frames):
        if self.frame_interval is None:
            interval = rec_no_frames / float(self.num_frames)
        else:
            interval = self.frame_interval
        valid_sta_range = rec_no_frames - (self.num_frames - 1) * interval
        if self.resnet3d_test_sample or self.HMDB_sample:
            valid_sta_range = rec_no_frames - self.num_frames * interval
        return interval, valid_sta_range

    def _build_bins_for_vds(self):
        self.video_bins = []
        self.bin_curr_idx = []
        self.video_index_offset = []
        curr_index_offset = 0

        for record in self.video_list:
            rec_no_frames = int(record.num_frames)
            _, valid_sta_range = self.__get_interval_valid_range(
                rec_no_frames)
            curr_num_bins = np.ceil(valid_sta_range * 1.0 / self.bin_interval)
            curr_num_bins = int(curr_num_bins)
            curr_bins = [
                (_idx,
                 (self.bin_interval * _idx,
                  min(self.bin_interval * (_idx + 1),
                      valid_sta_range)))
                for _idx in range(curr_num_bins)]

            self.video_bins.append(curr_bins)
            self.bin_curr_idx.append(0)
            self.video_index_offset.append(curr_index_offset)
            np.random.shuffle(self.video_bins[-1])

            curr_index_offset += curr_num_bins
        return curr_index_offset

    def _build_trn_bins(self):
        num_bins = self.trn_num_frames
        half_sec_frames = 12
        all_vds_bin_sta_end = []
        for record in self.video_list:
            rec_no_frames = int(record.num_frames)
            if self.HMDB_sample:
                needed_frames = self.trn_num_frames
                if rec_no_frames <= needed_frames:
                    rec_length = int(record.num_frames)
                    rec_no_frames = rec_length * (needed_frames//rec_length+1)
            
            frame_each_bin = min(half_sec_frames, rec_no_frames // num_bins)

            if frame_each_bin == 0:
                all_vds_bin_sta_end.append([])
                continue

            curr_bin_sta_end = []
            for curr_sta in range(0, rec_no_frames, frame_each_bin):
                curr_bin_sta_end.append(
                    (curr_sta,
                     min(curr_sta + frame_each_bin, rec_no_frames)))
            assert len(curr_bin_sta_end) >= num_bins
            all_vds_bin_sta_end.append(curr_bin_sta_end)

        self.all_vds_bin_sta_end = all_vds_bin_sta_end

    def _parse_list(self):
        # check the frame number is >= MIN_NUM_FRAMES
        # usualy it is [video_id, num_frames, class_idx]
        with open(self.metafile) as f:
            lines = [x.strip().split(' ') for x in f]
            lines = [line for line in lines
                     if int(line[1]) >= self.MIN_NUM_FRAMES]

        self.video_list = [VideoRecord(*v) for v in lines]
        if self.part_vd is not None:
            np.random.seed(0)
            now_len = len(self.video_list)
            chosen_indx = sorted(np.random.choice(
                range(now_len), int(now_len * self.part_vd)))
            self.video_list = [self.video_list[_tmp_idx]
                               for _tmp_idx in chosen_indx]

        print('Number of videos: {}'.format(len(self.video_list)))
        if self.bin_interval is not None:
            num_bins = self._build_bins_for_vds()
            print('Number of bins: {}'.format(num_bins))

        if self.trn_style:
            self._build_trn_bins()

    def _get_indices(self, record):
        rec_no_frames = int(record.num_frames)
        # For HMDB sampling, we'll loop the short videos to get enough frames.
        if self.HMDB_sample:
            if self.frame_interval is not None or self.trn_style:
                if not self.trn_style:
                    needed_frames = self.num_frames * self.frame_interval
                else:
                    needed_frames = self.trn_num_frames
                if rec_no_frames <= needed_frames:
                    rec_length = int(record.num_frames)
                    rec_no_frames = rec_length * (needed_frames//rec_length+1)
        # For 3D ResNet tesing, we need to get 10 non-overlapping clips.
        # So we first loop those short videos to get enough frames (16*10)
        if self.resnet3d_test_sample:
            needed_frames = self.num_frames * self.frame_interval * self.sample_groups
            if rec_no_frames <= needed_frames:
                rec_length = int(record.num_frames)
                rec_no_frames = needed_frames
        
        interval, valid_sta_range = self.__get_interval_valid_range(
            rec_no_frames)

        all_offsets = None
        for curr_start_group in range(self.sample_groups):
            # For tesing
            if self.frame_start is None:
                start_interval = valid_sta_range / (1.0 + self.sample_groups)
                sta_idx = start_interval * (curr_start_group + 1)
                if self.resnet3d_test_sample or self.HMDB_sample:
                    assert self.sample_groups > 1, 'Sample group should be greater than 1 during testing'
                    start_interval = valid_sta_range / (self.sample_groups-1)
                    sta_idx = start_interval * curr_start_group
            # For training
            elif self.frame_start == 'RANDOM':
                sta_idx = np.random.randint(valid_sta_range)
            
            if self.HMDB_sample or self.resnet3d_test_sample:
                rec_length = int(record.num_frames)
                offsets = np.array([int(sta_idx + interval * x)%rec_length
                                for x in range(self.num_frames)])
            else:
                offsets = np.array([int(sta_idx + interval * x)
                                for x in range(self.num_frames)])
            if all_offsets is None:
                all_offsets = offsets
            else:
                all_offsets = np.concatenate([all_offsets, offsets])
        return all_offsets + 1

    def _get_binned_indices(self, index, record):
        # NOTE: Have not implement HMDB_sampling for binned sampling
        rec_no_frames = int(record.num_frames)
        interval, valid_sta_range = self.__get_interval_valid_range(
            rec_no_frames)

        _bin_curr_idx = self.bin_curr_idx[index]
        _idx, (_sta_random,
               _end_random) = self.video_bins[index][_bin_curr_idx]
        assert self.frame_start == 'RANDOM', "Binned only supports random!"

        sta_idx = np.random.randint(_sta_random, _end_random)
        offsets = np.array([int(sta_idx + interval * x)
                            for x in range(self.num_frames)])
        self.bin_curr_idx[index] += 1
        if self.bin_curr_idx[index] == len(self.video_bins[index]):
            self.bin_curr_idx[index] = 0
            np.random.shuffle(self.video_bins[index])
        return offsets + 1, _idx + self.video_index_offset[index]

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

        # For HMDB and 3dresnet testing, we'll loop the short videos to get 
        # enough frames instead of changing to another video.
        if self.HMDB_sample or self.resnet3d_test_sample:
            return record, index

        if self.frame_interval is not None or self.trn_style:
            needed_frames = self.get_needed_frames()
            while int(record.num_frames) <= needed_frames:
                index = np.random.randint(len(self.video_list))
                record = self.video_list[index]
        return record, index

    def get_needed_frames(self):
        if not self.trn_style:
            needed_frames = self.num_frames * self.frame_interval
        else:
            needed_frames = self.trn_num_frames
        return needed_frames

    def _get_TRN_style_indices(self, index, record):
        curr_bin = self.all_vds_bin_sta_end[index]

        valid_sta_range = len(curr_bin) - self.trn_num_frames + 1
        all_offsets = None
        start_interval = int(valid_sta_range / (1.0 + self.sample_groups))
        rec_length = int(record.num_frames)
        for curr_start_group in range(self.sample_groups):
            if self.frame_start is None:
                sta_idx = start_interval * (curr_start_group + 1)
                offsets = np.array([int(np.mean(curr_bin[sta_idx + x]))
                                    for x in range(self.trn_num_frames)])
                if self.HMDB_sample:
                    offsets = np.array([int(np.mean(curr_bin[sta_idx + x]))%rec_length \
                                    for x in range(self.trn_num_frames)])
            elif self.frame_start == 'RANDOM':
                sta_idx = np.random.randint(valid_sta_range)
                offsets = np.array([np.random.randint(*curr_bin[sta_idx + x])
                                    for x in range(self.trn_num_frames)])
                if self.HMDB_sample:
                    offsets = np.array([np.random.randint(*curr_bin[sta_idx + x])%rec_length \
                                    for x in range(self.trn_num_frames)])

            # Seems this line is redundant
            #offsets = np.array([np.random.randint(*curr_bin[sta_idx + x]) \
            #                    for x in range(self.trn_num_frames)])

            if all_offsets is None:
                all_offsets = offsets
            else:
                all_offsets = np.concatenate([all_offsets, offsets])
        return all_offsets + 1

    def _get_indices_and_instance_index(self, index, record):
        if self.bin_interval is None:
            if not self.trn_style:
                indices = self._get_indices(record)
            else:
                indices = self._get_TRN_style_indices(index, record)
            vd_instance_index = index
        else:
            indices, vd_instance_index = self._get_binned_indices(
                index, record)
        #print(self.frame_start, int(record.num_frames), indices)
        return indices, vd_instance_index

    def __getitem__(self, index):
        record, index = self._get_valid_video(index)

        indices, vd_instance_index = self._get_indices_and_instance_index(
            index, record)

        frames = self.transform([self._load_image(record.path, int(idx))
                                 for idx in indices])
        return frames, int(record.label), vd_instance_index

    def __len__(self):
        return len(self.video_list)


def range_avg_step(start, stop=None, step=1.0, nout=None,
                   max_retries=1000):
    """Create a range of ints where the expected step size is a float `step`.
    """
    assert not (stop is None and nout is None)
    assert not (nout is not None and stop is not None)
    if nout is None:
        nout = int((stop - start) / step)
    elif stop is None:
        stop = int(start + nout * step)

    def _range_avg_step(start, stop, step):
        step, frac = divmod(step, 1)
        i = 0
        out = []
        x = range(start, stop)
        while i < len(x):
            out.append(x[i])
            s = int(step) + (random.random() < frac)
            i += s
        return out

    for _ in range(max_retries):
        out = _range_avg_step(start, stop, step)
        if len(out) == nout:
            return out
    else:
        raise ValueError('Could not find desired range.')


class RotVideoDataset(VideoDataset):
    def __init__(
            self, root, metafile,
            num_frames=8, frame_interval=None, frame_start=None,
            file_tmpl='{:06d}.jpg', transform=None, sample_groups=1,
            bin_interval=None,
            trn_style=False, trn_num_frames=8,
            part_vd=None,
            fps_conversion_factor=1.0,  # source_fps / desired_fps
            HMDB_sample=False, resnet3d_test_sample=False
    ):

        self.root = root
        self.metafile = metafile
        self.file_tmpl = file_tmpl
        self.transform = transform
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.frame_start = frame_start
        self.sample_groups = sample_groups
        self.bin_interval = bin_interval
        self.trn_style = trn_style
        self.trn_num_frames = trn_num_frames
        self.part_vd = part_vd
        self.fps_conversion_factor = fps_conversion_factor
        self.HMDB_sample = HMDB_sample
        self.resnet3d_test_sample = resnet3d_test_sample

        self._parse_list()

    def get_needed_frames(self):
        needed_frames = self.num_frames \
                * self.frame_interval \
                * self.fps_conversion_factor
        needed_frames = np.ceil(needed_frames)
        return needed_frames

    def __get_interval_valid_range(self, rec_no_frames):
        if self.frame_interval is None:
            interval = rec_no_frames / float(self.num_frames)
        else:
            interval = self.frame_interval
        valid_sta_range = rec_no_frames \
                - (self.num_frames - 1) \
                  * (interval * self.fps_conversion_factor)
        return interval, valid_sta_range

    def _get_indices(self, record):
        rec_no_frames = int(record.num_frames)
        interval, valid_sta_range = self.__get_interval_valid_range(
            rec_no_frames)

        all_offsets = None
        start_interval = valid_sta_range / (1.0 + self.sample_groups)
        for curr_start_group in range(self.sample_groups):
            if self.frame_start is None:
                sta_idx = start_interval * (curr_start_group + 1)
            elif self.frame_start == 'RANDOM':
                sta_idx = np.random.randint(valid_sta_range)
            if self.fps_conversion_factor != 1.0:
                offsets = np.array(range_avg_step(
                    int(sta_idx),
                    stop=None,
                    step=(interval * self.fps_conversion_factor),
                    nout=self.num_frames))
            else:
                offsets = np.array([int(sta_idx + interval * x)
                                    for x in range(self.num_frames)])
            if all_offsets is None:
                all_offsets = offsets
            else:
                all_offsets = np.concatenate([all_offsets, offsets])
        return all_offsets + 1

    def __getitem__(self, index):
        record, index = self._get_valid_video(index)

        indices, vd_instance_index = self._get_indices_and_instance_index(
            index, record)

        frames = self.transform([self._load_image(record.path, int(idx))
                                 for idx in indices])

        return frames, int(record.label), vd_instance_index


class VideoCollageDataset(VideoDataset):
    extensions = [
        '.jpg',
        '.jpeg',
        '.png',
        '.ppm',
        '.bmp',
        '.pgm',
        '.tif',
        '.JPEG']

    def _parse_list(self):
        self.video_list = []
        with open(self.metafile) as f:
            for line in f.readlines():
                #filename, num_frames, size, label = line.strip().split(' ')
                filename, num_frames, im_h, im_w, label = line.strip().split()
                size = (im_h, im_w)
                label = int(label)
                num_frames = int(num_frames)
                size = map(int, size)
                path = os.path.join(self.root, filename + '.jpg')
                if any(path.endswith(ext) for ext in self.extensions):
                    self.video_list.append(
                        VideoCollageRecord(
                            path, num_frames,
                            size, label))
        print('Number of videos: {}'.format(len(self.video_list)))

    def _load_collage(
            self, filename, size, indices,
            num_frames=None, nrow=8):
        im_h, im_w = size
        collage = Image.open(filename).convert('RGB')
        collage_w, collage_h = collage.size

        images = []
        num_images = 0
        for pos_y in range(0, collage_h, im_h):
            for pos_x in range(0, collage_w, im_w):
                num_images += 1
                if num_images in indices:
                    area = (pos_x, pos_y, pos_x + im_w, pos_y + im_h)
                    image = collage.crop(area)
                    images.append(image)

        return images

    def __getitem__(self, index):
        record = self.video_list[index]

        if self.frame_interval is not None:
            needed_frames = self.num_frames * self.frame_interval
            while int(record.num_frames) <= needed_frames:
                index = np.random.randint(len(self.video_list))
                record = self.video_list[index]
        if self.bin_interval is None:
            indices = self._get_indices(record)
            vd_instance_index = index
        else:
            indices, vd_instance_index = self._get_binned_indices(
                index, record)

        frames = self._load_collage(record.path, record.size, indices)
        if self.transform is not None:
            frames = self.transform(frames)
        if frames.shape[1] < len(indices):
            print(frames.shape[1], len(indices),
                  indices, record.size,
                  Image.open(record.path).convert('RGB').size)
            print(record.path)
        return frames, int(record.label), vd_instance_index

    def __len__(self):
        return len(self.video_list)


def test_get_vd_jpg_dataset():
    import config
    import transforms
    import socket

    hostname = socket.gethostname()

    if 'ccn' in hostname:
        root = '/mnt/fs3/chengxuz/kinetics/pt_meta'
        root_data = '/data5/chengxuz/Dataset/kinetics/comp_jpgs_extracted'
        cfg = config.dataset_config('kinetics', root=root, root_data=root_data)
    else:
        cfg = config.dataset_config('kinetics')

    transform = transforms.video_transform_color()

    # dataset = VideoDataset(
    #        cfg['root'], cfg['train_metafile'],
    #        num_frames=16, frame_interval=4, transform=transform,
    #        frame_start='RANDOM')

    # dataset = VideoDataset(
    #    cfg['root'], cfg['train_metafile'],
    #    num_frames=1, transform=transform,
    #    frame_start='RANDOM',
    #    bin_interval=52)

    dataset = VideoDataset(
        cfg['root'], cfg['train_metafile'],
        trn_style=True, transform=transform,
        frame_start='RANDOM',
        trn_num_frames=4)
    return dataset


def test_get_vd_collage_dataset():
    import config
    import transforms
    import socket

    root_data = '/mnt/fs3/chengxuz/kinetics/collages_extracted'
    meta_path = '/mnt/fs3/chengxuz/kinetics/pt_meta/train_collage_meta_all.txt'

    transform = transforms.video_transform_color()
    dataset = VideoCollageDataset(
        root_data, meta_path,
        num_frames=16, frame_interval=4,
        transform=transform,
        frame_start='RANDOM')
    return dataset


def test_rot3d_video_dataset():
    import config
    import transforms
    #root_data = '/data/vision/oliva/scratch/datasets/kinetics/comp_jpgs_extracted'
    #_, meta_path, _, root_data, _ = config.return_kinetics()
    root = '/mnt/fs3/chengxuz/kinetics/pt_meta'
    root_data = '/data5/chengxuz/Dataset/kinetics/comp_jpgs_extracted'
    cfg = config.dataset_config(
            'kinetics', root=root, root_data=root_data)

    transform = transforms.video_3DRot_transform()
    fps_conversion_factor = (25 / 16)
    dataset = RotVideoDataset(
        #root_data, meta_path,
        cfg['root'], cfg['train_metafile'],
        num_frames=16, frame_interval=1,
        transform=transform,
        frame_start='RANDOM',
        fps_conversion_factor=fps_conversion_factor
    )
    return dataset


if __name__ == '__main__':
    import torch

    # dataset = test_get_vd_jpg_dataset()
    #dataset = test_get_vd_collage_dataset()
    dataset = test_rot3d_video_dataset()

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True,
        num_workers=30, pin_memory=False,
        worker_init_fn=lambda x: np.random.seed(x))

    curr_time = time.time()
    init_time = curr_time
    data_enumerator = enumerate(dataloader)
    for i in range(100):
        _, (input, target, index) = data_enumerator.__next__()
        print(input.shape, input.dtype, np.max(input.numpy()))
        curr_time = time.time()
    print(time.time() - init_time)
