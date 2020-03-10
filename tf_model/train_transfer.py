from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np
import tensorflow as tf
import pdb

import json
import copy
import argparse
import time
import functools
import inspect

from model import preprocessing as prep
from model import instance_model
from model.dataset_utils import dataset_func
from utils import DATA_LEN_IMAGENET_FULL
import utils
import config


def get_config():
    cfg = config.Config()
    cfg.add('batch_size', type=int, default=128,
            help='Training batch size')
    cfg.add('test_batch_size', type=int, default=64,
            help='Testing batch size')
    cfg.add('init_lr', type=float, default=0.01,
            help='Initial learning rate')
    cfg.add('gpu', type=str, required=True,
            help='Value for CUDA_VISIBLE_DEVICES')
    cfg.add('weight_decay', type=float, default=1e-4,
            help='Weight decay')
    # TODO: Merge the 2 dataset-related parameters
    cfg.add('infant_data', type=bool, default=False,
            help='Whether use infant_headcam data for testing')
    cfg.add('dataset', type=str, default=None,
            help='The name of the dataset')
    cfg.add('train_len', type=int, default=utils.DATA_LEN_KINETICS_400,
            help='The length of the training data, when specifying dataset')
    cfg.add('val_len', type=int, default=utils.VAL_DATA_LEN_KINETICS_400,
            help='The length of the validation data, when specifying dataset')
    cfg.add('num_classes', type=int, default=1000,
            help='Number of classes')                
    cfg.add('image_dir', type=str, required=True,
            help='Directory containing dataset')
    cfg.add('metafile_root', type=str, default=None,
            help='Directory containing metafiles')
    cfg.add('q_cap', type=int, default=102400,
            help='Shuffle queue capacity of tfr data')
    cfg.add('ten_crop', type=bool,
            help='Whether do ten crop validation')
    cfg.add('num_tile', type=int, default=None,
            help='Whether tiling input images in data provider')

    # Loading parameters
    cfg.add('load_exp', type=str, required=True,
            help='The experiment to load from, in the format '
                 '[dbname]/[collname]/[exp_id]')
    cfg.add('load_step', type=int, default=None,
            help='Step number for loading')
    cfg.add('load_port', type=int,
            help='Port number of mongodb for loading (defaults to saving port')
    cfg.add('resume', type=bool,
            help='Flag for loading from last step of this exp_id, will override'
            ' all other loading options.')
    cfg.add('from_ckpt', default=None, type=str,
            help='Checkpoint path to directly load from')
    cfg.add('load_param_dict', default=None, type=dict,
            help='Specify part of the variables to be restored')

    # Saving parameters
    cfg.add('port', type=int, required=True,
            help='Port number for mongodb')
    cfg.add('host', type=str, default='localhost',
            help='Host for mongodb')
    cfg.add('save_exp', type=str, required=True,
            help='The [dbname]/[collname]/[exp_id] of this experiment.')
    cfg.add('cache_dir', type=str, required=True,
            help='Prefix of cache directory for tfutils')
    cfg.add('fre_valid', type=int, default=10009,
            help='Frequency of validation')
    cfg.add('fre_metric', type=int, default=1000,
            help='Frequency of saving metrics')
    cfg.add('fre_filter', type=int, default=10009,
            help='Frequency of saving filters')
    cfg.add('fre_cache_filter', type=int,
            help='Frequency of caching filters')

    # Training parameters
    cfg.add('model_type', type=str, default='single_frame',
            help='Model type, resnet or alexnet')
    cfg.add('resnet_size', type=int, default=18,
            help='ResNet size')
    cfg.add('get_all_layers', type=str, default=None,
            help='Whether get outputs for all layers')
    cfg.add('lr_boundaries', type=str, default=None,
            help='Learning rate boundaries for 10x drops')
    cfg.add('train_crop', type=str, default='default',
            help='Train crop style')

    cfg.add('tfutils', type=bool,
            help='Whether using tfutils')
    return cfg


def reg_loss_in_tfutils(loss, which_device, weight_decay):
    # Add weight decay to the loss.
    model_prefix = '/'.join(
            tf.get_default_graph().get_name_scope().split('/')[:2])
    def exclude_batch_norm_and_other_device(name):
        return 'batch_normalization' not in name \
                and name.startswith(model_prefix) \
                and 'instance' in name
    l2_loss = weight_decay * tf.add_n(
            [tf.nn.l2_loss(tf.cast(v, tf.float32))
                for v in tf.trainable_variables()
                if exclude_batch_norm_and_other_device(v.name)])
    loss_all = tf.add(loss, l2_loss)
    return loss_all


def reg_loss(loss, which_device, weight_decay):
    # Add weight decay to the loss.
    def exclude_batch_norm(name):
        return 'batch_normalization' not in name \
                and 'instance' in name
    l2_loss = weight_decay * tf.add_n(
            [tf.nn.l2_loss(tf.cast(v, tf.float32))
                for v in tf.trainable_variables()
                if exclude_batch_norm(v.name)])
    loss_all = tf.add(loss, l2_loss)
    return loss_all


def tfutils_func_params(func, to_record, **kwargs):
    '''
    Helper for creating parameters describing a function to be passed
    to tfutils.
    '''
    for k in to_record:
        if k not in kwargs:
            raise Exception("Cannot record parameter %r which "
                            "does not appear in kwargs." % k)

    params, partial_kwargs = {}, {}
    for k, v in kwargs.items():
        if k in to_record:
            # These parameters will be recorded by tfutils.
            params[k] = v
        else:
            partial_kwargs[k] = v

    # Similar to functools.partial, but we do it this way to retain
    # some module info (which tfutils assumes exists).
    @functools.wraps(func)
    def partial_func(**kw):
        kw.update(partial_kwargs)
        return func(**kw)
    assert 'func' not in params
    params['func'] = partial_func
    return params


def add_training_params(params, args):
    NUM_BATCHES_PER_EPOCH = DATA_LEN_IMAGENET_FULL / args.batch_size

    # model_params: a function that will build the model
    model_params = {
        'func': instance_model.build_transfer_output,
        'trainable_scopes': ['instance'],
        'get_all_layers': args.get_all_layers,
        "model_type": args.model_type,
        "resnet_size": args.resnet_size,
        "num_classes": args.num_classes,
    }
    multi_gpu = len(args.gpu.split(','))
    if multi_gpu > 1 or args.gpu_offset != 0:
        model_params['num_gpus'] = multi_gpu
        model_params['devices'] = [
                '/gpu:%i' % (idx + args.gpu_offset) \
                for idx in range(multi_gpu)]
    params['model_params'] = model_params

    # train_params: parameters about training data
    process_img_func = prep.resnet_train
    if args.train_crop == 'resnet_crop_flip':
        process_img_func = prep.resnet_crop_flip
    elif args.train_crop == 'alexnet_crop_flip':
        process_img_func = prep.alexnet_crop_flip
    elif args.train_crop == 'validate_crop':
        process_img_func = prep.resnet_validate
    elif args.train_crop == 'outshape_112':
        process_img_func = prep.resnet_train_112

    train_data_param = tfutils_func_params(
        dataset_func, ['image_dir', 'is_train', 'q_cap', 'batch_size'],
        image_dir=args.image_dir,
        process_img_func=process_img_func,
        is_train=True,
        q_cap=args.q_cap,
        batch_size=args.batch_size,
        num_tile=args.num_tile,
    )

    def _train_target_func(
        inputs,
        output,
        get_all_layers=None,
        *args,
        **kwargs):
        if not get_all_layers:
            return {'accuracy': output[1]}
        else:
            return {'accuracy': tf.reduce_mean(output[1].values())}

    params['train_params'] = {
        'validate_first': False,
        'data_params': train_data_param,
        'queue_params': None,
        'thres_loss': float('Inf'),
        'num_steps': 2000 * NUM_BATCHES_PER_EPOCH,
        'targets': {
                'func': _train_target_func,
                'get_all_layers': args.get_all_layers,
                },
    }

    # loss_params: parameters to build the loss
    def loss_func(output, *args, **kwargs):
        return output[0]
    params['loss_params'] = {
        'pred_targets': [],
        # we don't want GPUs to calculate l2 loss separately
        'agg_func': reg_loss,
        'agg_func_kwargs': {'weight_decay': args.weight_decay},
        'loss_func': loss_func,
    }


def add_validation_params(params, args):
    # validation_params: control the validation
    val_len = 50000
    valid_prep_func = prep.resnet_validate
    if args.train_crop == 'outshape_112':
        valid_prep_func = prep.resnet_validate_112
    if args.ten_crop:
        assert args.num_tile is None
        valid_prep_func = prep.resnet_10crop_validate

    topn_val_data_param = tfutils_func_params(
        dataset_func, ['image_dir', 'q_cap', 'batch_size'],
        image_dir=args.image_dir,
        process_img_func=valid_prep_func,
        is_train=False,
        q_cap=args.test_batch_size,
        batch_size=args.test_batch_size,
        num_tile=args.num_tile,
    )

    def online_agg(agg_res, res, step):
        if agg_res is None:
            agg_res = {k: [] for k in res}
        for k, v in res.items():
            agg_res[k].append(np.mean(v))
        return agg_res
    def valid_perf_func(inputs, output):
        if not args.get_all_layers:
            return {'top1': output}
        else:
            ret_dict = {}
            for key, each_out in output.items():
                ret_dict['top1_{name}'.format(name=key)] = each_out
            return ret_dict

    topn_val_param = {
        'data_params': topn_val_data_param,
        'queue_params': None,
        'targets': {'func': valid_perf_func},
        # TODO: slight rounding error?
        'num_steps': int(val_len/args.test_batch_size),
        'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
        'online_agg_func': online_agg,
    }
    params['validation_params'] = {
        'topn': topn_val_param,
    }


def add_save_and_load_params(params, args):
    # save_params: defining where to save the models
    db_name, col_name, exp_id = args.save_exp.split('/')
    cache_dir = os.path.join(
            args.cache_dir, 'models',
            db_name, col_name, exp_id)
    params['save_params'] = {
        'host': 'localhost',
        'port': args.port,
        'dbname': db_name,
        'collname': col_name,
        'exp_id': exp_id,
        'do_save': True,
        'save_initial_filters': True,
        'save_metrics_freq': args.fre_metric,
        'save_valid_freq': args.fre_valid,
        'save_filters_freq': args.fre_filter,
        'cache_filters_freq': args.fre_cache_filter or args.fre_filter,
        'cache_dir': cache_dir,
    }

    # load_params: defining where to load, if needed
    if args.resume or args.load_exp is None:
        load_exp = args.save_exp
    else:
        load_exp = args.load_exp
    load_dbname, load_collname, load_exp_id = load_exp.split('/')
    if args.resume or args.load_step is None:
        load_query = None
    else:
        load_query = {
            'exp_id': load_exp_id,
            'saved_filters': True,
            'step': args.load_step
        }
    params['load_params'] = {
        'host': 'localhost',
        'port': args.load_port or args.port,
        'dbname': load_dbname,
        'collname': load_collname,
        'exp_id': load_exp_id,
        'do_restore': True,
        'query': load_query,
        'from_ckpt': args.from_ckpt,
    }
    if args.load_param_dict is not None:
        params['load_params']['load_param_dict'] = args.load_param_dict


def add_optimization_params(params, args):
    # learning_rate_params: build the learning rate
    # For now, just stay the same
    NUM_BATCHES_PER_EPOCH = DATA_LEN_IMAGENET_FULL / args.batch_size
    if args.dataset is not None:
        NUM_BATCHES_PER_EPOCH = args.train_len / args.batch_size
    params['learning_rate_params'] = {
        'func': tf.train.exponential_decay,
        'learning_rate': args.init_lr,
        'decay_rate': 1,
        'decay_steps': NUM_BATCHES_PER_EPOCH,
        'staircase': True
    }

    if args.lr_boundaries is not None:
        boundaries = args.lr_boundaries.split(',')
        boundaries = [int(each_boundary) for each_boundary in boundaries]

        all_lrs = [
                args.init_lr * (0.1 ** drop_level) \
                for drop_level in range(len(boundaries) + 1)]

        params['learning_rate_params'] = {
                'func':lambda global_step, boundaries, values:\
                        tf.train.piecewise_constant(
                            x=global_step,
                            boundaries=boundaries, values=values),
                'values': all_lrs,
                'boundaries': boundaries,
                }

    # optimizer_params
    params['optimizer_params'] = {
            'optimizer': tf.train.MomentumOptimizer,
            'momentum': .9,
            }


def get_all_params_from_arg(args):
    params = {
        'skip_check': True,
        'log_device_placement': False
    }

    add_training_params(params, args)
    add_save_and_load_params(params, args)
    add_optimization_params(params, args)
    add_validation_params(params, args)
    return params


def start_training(params, args):
    if args.tfutils:
        params['loss_params']['agg_func'] = reg_loss_in_tfutils
        db_name, col_name, exp_id = args.save_exp.split('/')
        cache_dir = os.path.join(
                args.cache_dir, 'models_tfutils',
                db_name, col_name, exp_id)
        params['save_params']['cache_dir'] = cache_dir
        from tfutils import base
        base.train_from_params(**params)
    else:
        from framework import TrainFramework
        train_framework = TrainFramework(params)
        train_framework.train()


def main():
    # Parse arguments
    cfg = get_config()
    args = cfg.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    params = get_all_params_from_arg(args)
    start_training(params, args)


if __name__ == "__main__":
    main()
