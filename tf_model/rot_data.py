from __future__ import division, print_function, absolute_import
import os, sys
import torch
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath('../'))
from pt_loader import transforms, datasets
import data
FPS_FACTOR = (25 / 16)


def get_train_rot_pt_loader(args):
    cfg, _ = data.get_cfg_transform(args)
    if not args.rot_real_prep:
        transform = transforms.video_3DRot_transform()
    else:
        transform = transforms.video_3DRot_transform_real_resize()
    dataset = datasets.RotVideoDataset(
        cfg['root'], cfg['train_metafile'],
        num_frames=16, frame_interval=1,
        transform=transform,
        frame_start='RANDOM',
        fps_conversion_factor=FPS_FACTOR)
    return data.get_train_dataloader(args, dataset)


def get_val_rot_pt_loader(args):
    cfg, _ = data.get_val_cfg_transform(args)
    if not args.rot_real_prep:
        transform = transforms.video_3DRot_transform_val()
    else:
        transform = transforms.video_3DRot_transform_val((136, 136))

    dataset = datasets.RotVideoDataset(
        cfg['root'], cfg['val_metafile'],
        num_frames=16, frame_interval=1,
        transform=transform,
        fps_conversion_factor=FPS_FACTOR)
    return data.get_val_dataloader(args, dataset)


def get_rot_placeholders(
        batch_size, 
        crop_size=112, num_channels=3,
        name_prefix='TRAIN'):
    num_frames = 64
    image_placeholder = tf.placeholder(
            tf.uint8, 
            (batch_size, num_frames, crop_size, crop_size, num_channels),
            name='%s_IMAGE_PLACEHOLDER' % name_prefix)
    inputs = {'image': image_placeholder}
    return inputs


def get_feeddict(image, name_prefix='TRAIN'):
    image_placeholder = tf.get_default_graph().get_tensor_by_name(
            '%s_IMAGE_PLACEHOLDER:0' % name_prefix)
    feed_dict = {image_placeholder: image[0].numpy()}
    return feed_dict
