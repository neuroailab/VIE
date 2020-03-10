from __future__ import division, print_function, absolute_import
import os, sys
import torch
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath('../'))
from pt_loader import datasets, config, transforms, opn_datasets


def get_feeddict(image, label, index, name_prefix='TRAIN'):
    image_placeholder = tf.get_default_graph().get_tensor_by_name(
            '%s_IMAGE_PLACEHOLDER:0' % name_prefix)
    label_placeholder = tf.get_default_graph().get_tensor_by_name(
            '%s_LABEL_PLACEHOLDER:0' % name_prefix)
    index_placeholder = tf.get_default_graph().get_tensor_by_name(
            '%s_INDEX_PLACEHOLDER:0' % name_prefix)
    feed_dict = {
            image_placeholder: image.numpy(),
            label_placeholder: label.numpy(),
            index_placeholder: index.numpy()}
    return feed_dict


def get_cfg_transform(args):
    root = '/mnt/fs3/chengxuz/kinetics/pt_meta'
    if args.metafile_root is not None:
        root = args.metafile_root
    root_data = args.image_dir
    """
    # TODO: May have problem later
    # TODO: Would be better to specify both train_dataset and test_dataset
    if args.only_emb:
        cfg = config.dataset_config('kinetics', 
                                    root='/mnt/fs3/chengxuz/kinetics/pt_meta', 
                                    root_data='/data5/chengxuz/Dataset/kinetics/comp_jpgs_extracted')
    else:"""
    cfg = config.dataset_config(args.dataset, root=root, root_data=root_data)

    if args.train_prep is None:
        transform = transforms.video_transform()
    elif args.train_prep == 'RandomSized':
        transform = transforms.video_transform_rdsz()
    elif args.train_prep == 'ColorJitter':
        transform = transforms.video_transform_color()
    elif args.train_prep == 'ColorJitter_112':
        transform = transforms.video_transform_color(
                frame_size_min=128, frame_size_max=160,
                crop_size=112)
    elif args.train_prep == 'ColorJitterRandomSized':
        transform = transforms.video_transform_color_rdsz()
    elif args.train_prep == '3DRotNet_finetune':
        transform = transforms.video_3DRot_finetune_transform()
    elif args.train_prep == 'MultiScaleCrop_112':
        transform = transforms.video_transform_multiscalecrop()
    elif args.train_prep == 'MultiScaleCrop_224':
        transform = transforms.video_transform_multiscalecrop(size=224)
    else:
        raise NotImplementedError('Preprocessing specified not implemented!')
    return cfg, transform


def get_train_dataloader(args, dataset):
    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, 
            num_workers=args.train_num_workers, pin_memory=False, 
            worker_init_fn=lambda x: np.random.seed(x))
    return dataloader


def get_train_pt_loader(args):
    cfg, transform = get_cfg_transform(args)

    dataset = datasets.VideoDataset(
            cfg['root'], cfg['train_metafile'], 
            num_frames=1, transform=transform, 
            frame_start='RANDOM',
            bin_interval=args.bin_interval,
            part_vd=args.part_vd,
            HMDB_sample=False, resnet3d_test_sample=False)
    return get_train_dataloader(args, dataset)


def get_val_cfg_transform(args):
    root = '/mnt/fs3/chengxuz/kinetics/pt_meta'
    if args.metafile_root is not None:
        root = args.metafile_root
    root_data = args.val_image_dir
    # cfg = config.dataset_config('kinetics', root=root, root_data=root_data)
    cfg = config.dataset_config(args.dataset, root=root, root_data=root_data)

    if args.train_prep is not None and '112' in args.train_prep:
        transform = transforms.video_transform_val(
                frame_size=128, crop_size=112)
    elif args.train_prep == "3DRotNet_finetune":
        transform = transforms.video_3DRot_finetune_val((136, 136))
    elif args.train_prep == 'MultiScaleCrop_112':
        transform = transforms.video_transform_multiscalecrop(scales=[1],
        crop_positions=['c'])
    elif args.train_prep == 'MultiScaleCrop_224':
        transform = transforms.video_transform_multiscalecrop(scales=[1],
        crop_positions=['c'], size=224)

    else:
        transform = transforms.video_transform_val(dataset=args.dataset)
    return cfg, transform


def get_val_dataloader(args, dataset):
    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.test_batch_size, shuffle=False, 
            num_workers=args.val_num_workers, pin_memory=False)
    return dataloader


def get_val_pt_loader(args):
    cfg, transform = get_val_cfg_transform(args)

    dataset = datasets.VideoDataset(
            cfg['root'], cfg['val_metafile'], 
            num_frames=args.test_no_frames, transform=transform,
            HMDB_sample=False, resnet3d_test_sample=False)
    return get_val_dataloader(args, dataset)


def get_train_slow_pt_loader(args):
    cfg, transform = get_cfg_transform(args)

    dataset = datasets.VideoDataset(
            cfg['root'], cfg['train_metafile'], 
            num_frames=4, frame_interval=16, 
            transform=transform, 
            frame_start='RANDOM',
            bin_interval=args.bin_interval,
            HMDB_sample=args.HMDB_sample)
    return get_train_dataloader(args, dataset)


def get_val_slow_pt_loader(args):
    cfg, transform = get_val_cfg_transform(args)

    dataset = datasets.VideoDataset(
            cfg['root'], cfg['val_metafile'], 
            num_frames=4, frame_interval=16, 
            sample_groups=args.test_no_frames, transform=transform,
            HMDB_sample=args.HMDB_sample)
    return get_val_dataloader(args, dataset)


def get_train_fast_pt_loader(args):
    cfg, transform = get_cfg_transform(args)

    dataset = datasets.VideoDataset(
            cfg['root'], cfg['train_metafile'], 
            num_frames=32, frame_interval=2, 
            transform=transform, 
            frame_start='RANDOM',
            bin_interval=args.bin_interval,
            HMDB_sample=args.HMDB_sample)
    return get_train_dataloader(args, dataset)


def get_val_fast_pt_loader(args):
    cfg, transform = get_val_cfg_transform(args)

    dataset = datasets.VideoDataset(
            cfg['root'], cfg['val_metafile'], 
            num_frames=32, frame_interval=2, 
            sample_groups=args.test_no_frames, transform=transform,
            HMDB_sample=args.HMDB_sample)
    return get_val_dataloader(args, dataset)


def get_train_fast_a4_pt_loader(args):
    cfg, transform = get_cfg_transform(args)

    dataset = datasets.VideoDataset(
            cfg['root'], cfg['train_metafile'], 
            num_frames=16, frame_interval=4, 
            transform=transform, 
            frame_start='RANDOM',
            bin_interval=args.bin_interval,
            HMDB_sample=True, resnet3d_test_sample=False)
    return get_train_dataloader(args, dataset)


def get_val_fast_a4_pt_loader(args):
    cfg, transform = get_val_cfg_transform(args)

    dataset = datasets.VideoDataset(
            cfg['root'], cfg['val_metafile'], 
            num_frames=16, frame_interval=4, 
            sample_groups=args.test_no_frames, transform=transform,
            HMDB_sample=False, resnet3d_test_sample=True)
    return get_val_dataloader(args, dataset)


def get_train_trn_pt_loader(args):
    cfg, transform = get_cfg_transform(args)

    dataset = datasets.VideoDataset(
            cfg['root'], cfg['train_metafile'], 
            trn_style=True, transform=transform, 
            frame_start='RANDOM',
            trn_num_frames=args.trn_num_frames,
            HMDB_sample=args.HMDB_sample)
    return get_train_dataloader(args, dataset)


def get_val_trn_pt_loader(args):
    cfg, transform = get_val_cfg_transform(args)

    dataset = datasets.VideoDataset(
            cfg['root'], cfg['val_metafile'], 
            trn_style=True, sample_groups=args.test_no_frames, 
            transform=transform,
            trn_num_frames=args.trn_num_frames,
            HMDB_sample=args.HMDB_sample)
    return get_val_dataloader(args, dataset)


def get_train_3dresnet_pt_loader(args):
    cfg, transform = get_cfg_transform(args)

    dataset = datasets.VideoDataset(
            cfg['root'], cfg['train_metafile'], 
            num_frames=16, frame_interval=1, 
            transform=transform, 
            frame_start='RANDOM',
            bin_interval=args.bin_interval,
            HMDB_sample=True, resnet3d_test_sample=False)
    if args.rotnet:
        FPS_FACTOR = (25 / 16)
        dataset = datasets.RotVideoDataset(
            cfg['root'], cfg['train_metafile'],
            num_frames=16, frame_interval=1,
            transform=transform,
            frame_start='RANDOM',
            fps_conversion_factor=FPS_FACTOR,
            HMDB_sample=False, resnet3d_test_sample=False)
    return get_train_dataloader(args, dataset)


def get_val_3dresnet_pt_loader(args):
    cfg, transform = get_val_cfg_transform(args)

    dataset = datasets.VideoDataset(
            cfg['root'], cfg['val_metafile'], 
            num_frames=16, frame_interval=1, 
            sample_groups=args.test_no_frames, transform=transform,
            HMDB_sample=False, resnet3d_test_sample=True)
    if args.rotnet:
        FPS_FACTOR = (25 / 16)
        dataset = datasets.RotVideoDataset(
            cfg['root'], cfg['val_metafile'],
            num_frames=16, frame_interval=1,
            sample_groups=args.test_no_frames, 
            transform=transform,
            fps_conversion_factor=FPS_FACTOR,
            HMDB_sample=False, resnet3d_test_sample=False)
    return get_val_dataloader(args, dataset)


def get_placeholders(
        batch_size, num_frames=1, 
        crop_size=224, num_channels=3,
        name_prefix='TRAIN', multi_frame=False, multi_group=None):
    image_placeholder = tf.placeholder(
            tf.uint8, 
            #(batch_size, num_channels, num_frames, crop_size, crop_size),
            (batch_size, num_frames, crop_size, crop_size, num_channels),
            name='%s_IMAGE_PLACEHOLDER' % name_prefix)
    label_placeholder = tf.placeholder(
            tf.int64,
            (batch_size),
            name='%s_LABEL_PLACEHOLDER' % name_prefix)
    index_placeholder = tf.placeholder(
            tf.int64,
            (batch_size),
            name='%s_INDEX_PLACEHOLDER' % name_prefix)
    if not multi_frame:
        if num_frames == 1:
            image_placeholder = tf.squeeze(image_placeholder, axis=1)
        else:
            image_placeholder = tf.reshape(
                    image_placeholder, 
                    [-1, crop_size, crop_size, num_channels])
    else:
        if multi_group is not None:
            image_placeholder = tf.reshape(
                    image_placeholder, 
                    [batch_size*multi_group, num_frames // multi_group, \
                            crop_size, crop_size, num_channels])
    inputs = {
            'image': image_placeholder,
            'label': label_placeholder,
            'index': index_placeholder}
    return inputs
