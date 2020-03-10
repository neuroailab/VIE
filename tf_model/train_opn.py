from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np
import tensorflow as tf
import cPickle

import json
import copy
import argparse
import time
import functools
import inspect
import pdb

from tfutils import base, optimizer
import tfutils.defaults

from model import opn_model

from utils import online_keep_all
import config
import opn_data

import train_vie
OPN_KINETICS_VIDEOS = 239871
OPN_KINETICS_VAL_VDS = 19647


def get_params_from_arg(args):
    save_params, load_params = train_vie.get_save_load_params_from_arg(args)
    loss_params, learning_rate_params, optimizer_params \
            = train_vie.get_loss_lr_opt_params_from_arg(args)

    # train_params
    train_data_loader = opn_data.get_train_opn_pt_loader(args)
    data_enumerator = [enumerate(train_data_loader)]
    def train_loop(sess, train_targets, num_minibatches=1, **params):
        assert num_minibatches==1, "Mini-batch not supported!"

        global_step_vars = [v for v in tf.global_variables() \
                            if 'global_step' in v.name]
        assert len(global_step_vars) == 1
        global_step = sess.run(global_step_vars[0])

        data_en_update_fre = OPN_KINETICS_VIDEOS // args.batch_size
        if global_step % data_en_update_fre == 0:
            data_enumerator.pop()
            data_enumerator.append(enumerate(train_data_loader))
        _, image = data_enumerator[0].next()
        feed_dict = opn_data.get_feeddict(image)
        sess_res = sess.run(train_targets, feed_dict=feed_dict)
        return sess_res

    train_data_param = {
            'func': opn_data.get_opn_placeholders,
            'batch_size': args.batch_size,
            'crop_size': args.opn_crop_size}
    train_params = {
            'validate_first': False,
            'data_params': train_data_param,
            'queue_params': None,
            'thres_loss': float('Inf'),
            'num_steps': float('Inf'),
            'train_loop': {'func': train_loop}}
    train_params['targets'] = {
            'func': lambda inputs, output: {'accuracy': output['accuracy']}}

    # validation_params
    val_len = 3 * OPN_KINETICS_VAL_VDS
    topn_val_data_param = {
            'func': opn_data.get_opn_placeholders,
            'batch_size': args.test_batch_size,
            'name_prefix': 'VAL',
            'crop_size': args.opn_crop_size}

    val_step_num = int(val_len / args.test_batch_size)
    val_data_loader = opn_data.get_val_opn_pt_loader(args)
    val_counter = [0]
    val_data_enumerator = [enumerate(val_data_loader)]
    def valid_loop(sess, target):
        val_counter[0] += 1
        if val_counter[0] % (OPN_KINETICS_VAL_VDS // args.test_batch_size) == 0:
            val_data_enumerator.pop()
            val_data_enumerator.append(enumerate(val_data_loader))
        _, image = val_data_enumerator[0].next()
        feed_dict = opn_data.get_feeddict(image, name_prefix='VAL')
        return sess.run(target, feed_dict=feed_dict)

    val_targets = {
            'func': lambda inputs, output: {'accuracy': output['accuracy']}}

    topn_val_param = {
        'data_params': topn_val_data_param,
        'queue_params': None,
        'targets': val_targets,
        'num_steps': val_step_num,
        'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
        'online_agg_func': train_vie.online_agg,
        'valid_loop': {'func': valid_loop}
        }
    validation_params = {'topn': topn_val_param}

    # model_params
    model_params = {
            'func': opn_model.build_loss_accuracy,
            'resnet_size': args.resnet_size}
    multi_gpu = len(args.gpu.split(','))
    if multi_gpu > 1:
        model_params['num_gpus'] = multi_gpu
        model_params['devices'] = ['/gpu:%i' % idx for idx in range(multi_gpu)]

    # Put all parameters together
    params = {
            'save_params': save_params,
            'load_params': load_params,
            'loss_params': loss_params,
            'learning_rate_params': learning_rate_params,
            'optimizer_params': optimizer_params,
            'log_device_placement': False,
            'skip_check': True,
            'train_params': train_params,
            'validation_params': validation_params,
            'model_params': model_params,
            }
    return params


def get_config():
    cfg = train_vie.get_config()
    cfg.add('opn_crop_size', type=int, default=80,
            help='Crop size for opn')
    cfg.add('opn_transform', type=str, default=None,
            help='Transform type for opn, None or Sep')
    cfg.add('opn_flow_folder', type=str, default=None,
            help='Not none, will use flow')
    return cfg


def main():
    # Parse arguments
    cfg = get_config()
    args = cfg.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Get params needed, start training
    params = get_params_from_arg(args)
    base.train_from_params(**params)


if __name__ == "__main__":
    main()
