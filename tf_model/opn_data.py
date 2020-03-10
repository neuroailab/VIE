from __future__ import division, print_function, absolute_import
import os, sys
import torch
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath('../'))
from pt_loader import transforms, opn_datasets
import data


def get_train_opn_pt_loader(args):
    cfg, _ = data.get_cfg_transform(args)
    transform = transforms.video_OPN_transform_color(
            crop_size=args.opn_crop_size)
    if args.opn_transform == 'Sep':
        transform = transforms.video_OPN_transform_sep_color(
                crop_size=args.opn_crop_size)

    if args.opn_flow_folder is None:
        dataset = opn_datasets.OPNVideoDataset(
                cfg['root'], cfg['train_metafile'], 
                transform=transform)
    else:
        dataset = opn_datasets.MotionAwareOPNVideoDataset(
                cfg['root'], args.opn_flow_folder, 
                cfg['train_metafile'], 
                transform=transform)
    return data.get_train_dataloader(args, dataset)


def get_val_opn_pt_loader(args):
    cfg, _ = data.get_val_cfg_transform(args)
    transform = transforms.video_transform_val(crop_size=args.opn_crop_size)

    if args.opn_flow_folder is None:
        dataset = opn_datasets.OPNVideoDataset(
                cfg['root'], cfg['val_metafile'], 
                transform=transform)
    else:
        dataset = opn_datasets.MotionAwareOPNVideoDataset(
                cfg['root'], args.opn_flow_folder, 
                cfg['val_metafile'], 
                transform=transform)
    return data.get_val_dataloader(args, dataset)


def get_opn_placeholders(
        batch_size, 
        crop_size=80, num_channels=3,
        name_prefix='TRAIN'):
    num_frames = 4
    image_placeholder = tf.placeholder(
            tf.uint8, 
            (batch_size, num_frames, crop_size, crop_size, num_channels),
            name='%s_IMAGE_PLACEHOLDER' % name_prefix)
    inputs = {'image': image_placeholder}
    return inputs


def get_feeddict(image, name_prefix='TRAIN'):
    image_placeholder = tf.get_default_graph().get_tensor_by_name(
            '%s_IMAGE_PLACEHOLDER:0' % name_prefix)
    feed_dict = {image_placeholder: image.numpy()}
    return feed_dict
