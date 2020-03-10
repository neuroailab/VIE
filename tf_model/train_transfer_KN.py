from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np
import tensorflow as tf

import json
import copy
import argparse
import time
import functools
import inspect

from model import instance_model
import train_transfer as prev_trans
import train_vie
import data


def get_config():
    cfg = prev_trans.get_config()
    cfg.add('finetune_conv', type=bool, default=False,
            help='Whether to finetune conv layers or not')        
    cfg.add('dropout', type=float, default=None,
            help='If not none, apply dropout at the given rate')        
    cfg.add('optimizer', type=str, default=None,
            help='If not none, use the given optimizer instead of momentum')        
    cfg.add('test_no_frames', type=int, default=5,
            help='Number of frames in one video during validation')
    cfg.add('train_prep', type=str, default=None,
            help='Train preprocessing')
    cfg.add('val_image_dir', type=str, required=True,
            help='Directory containing dataset')
    cfg.add('bin_interval', type=int, default=None,
            help='Bin interval for binned video dataset')
    cfg.add('part_vd', type=float, default=None,
            help='Portion of videos to use during training')
    cfg.add('trn_num_frames', type=int, default=8,
            help='Number of frames in trn style')
    cfg.add('HMDB_sample', type=bool, default=False,
            help='Whether to use HMDB sampling strategy')
    cfg.add('final_pooling', type=int, default=None,
            help='The output feature map size of the final pooling layer')
    cfg.add('slowfast_single_pooling', type=bool, default=False,
            help='Whether to add reduce mean final pooling for slowfast_single model')
    cfg.add('rotnet', type=bool, default=False,
            help='Whether finetuing 3D RotNet')
    cfg.add('train_num_workers', type=int, default=12,
            help='Training worker number')
    cfg.add('val_num_workers', type=int, default=12,
            help='Validation worker number')
    return cfg


def valid_func(
        inputs, output, 
        test_no_frames):
    def _get_one_top(output):
        num_classes = output.get_shape().as_list()[-1]
        curr_output = tf.nn.softmax(output)
        curr_output = tf.reshape(curr_output, [-1, test_no_frames, num_classes])
        curr_output = tf.reduce_mean(curr_output, axis=1)

        top1_accuracy = tf.nn.in_top_k(curr_output, inputs['label'], k=1)
        top5_accuracy = tf.nn.in_top_k(curr_output, inputs['label'], k=5)
        #return {'pred': curr_output, 'top1': top1_accuracy, 'top5': top5_accuracy}
        return {'top1': top1_accuracy, 'top5': top5_accuracy}

    if isinstance(output, dict):
        ret_dict = {}
        for key, _output in output.items():
            _one_ret_dict = _get_one_top(_output)
            ret_dict['top1_%s' % key] = _one_ret_dict['top1']
            ret_dict['top5_%s' % key] = _one_ret_dict['top5']
    else:
        ret_dict = _get_one_top(output)
    return ret_dict


def main():
    # Parse arguments
    cfg = get_config()
    args = cfg.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    params = {
        'skip_check': True,
        'log_device_placement': False
    }

    prev_trans.add_save_and_load_params(params, args)
    prev_trans.add_optimization_params(params, args)

    params['loss_params'] = {
        'pred_targets': [],
        'agg_func': prev_trans.reg_loss,
        'agg_func_kwargs': {'weight_decay': args.weight_decay},
        'loss_func': lambda output, *args, **kwargs: output['loss'],
    }

    # model_params
    model_params = {
            'func': instance_model.build_KN_transfer_output,
            'finetune_conv': args.finetune_conv,
            'get_all_layers': args.get_all_layers,
            "model_type": args.model_type,
            "resnet_size": args.resnet_size,
            'num_classes': args.num_classes,
            'dropout': args.dropout,
            "final_pooling": args.final_pooling,
            "slowfast_single_pooling": args.slowfast_single_pooling,
            }
    # Only train the readout layer
    if not args.finetune_conv:
        model_params['trainable_scopes'] = ['instance']
        
    multi_gpu = len(args.gpu.split(','))
    if multi_gpu > 1:
        model_params['num_gpus'] = multi_gpu
        model_params['devices'] = ['/gpu:%i' % idx for idx in range(multi_gpu)]
    params['model_params'] = model_params

    # train_params
    train_data_loader = train_vie.get_train_pt_loader_from_arg(args)
    data_enumerator = [enumerate(train_data_loader)]
    def train_loop(sess, train_targets, num_minibatches=1, **params):
        assert num_minibatches==1, "Mini-batch not supported!"

        global_step_vars = [v for v in tf.global_variables() \
                            if 'global_step' in v.name]
        assert len(global_step_vars) == 1
        global_step = sess.run(global_step_vars[0])

        # data_en_update_fre = train_vie.NUM_KINETICS_VIDEOS // args.batch_size
        data_en_update_fre = args.train_len // args.batch_size
        if global_step % data_en_update_fre == 0:
            data_enumerator.pop()
            data_enumerator.append(enumerate(train_data_loader))
        _, (image, label, index) = next(data_enumerator[0])
        feed_dict = data.get_feeddict(image, label, index)
        sess_res = sess.run(train_targets, feed_dict=feed_dict)
        return sess_res

    train_data_param = train_vie.get_train_data_param_from_arg(args)
    train_params = {
            'validate_first': False,
            'data_params': train_data_param,
            'queue_params': None,
            'thres_loss': float('Inf'),
            'num_steps': float('Inf'),
            'train_loop': {'func': train_loop},
            }
    train_params['targets'] = {
            'func': lambda inputs, output: {'accuracy': output['accuracy']}}
    params['train_params'] = train_params

    # validation_params
    # val_len = 19653
    topn_val_data_param = train_vie.get_topn_val_data_param_from_arg(args)
    valid_loop, val_step_num = train_vie.get_valid_loop_from_arg(args)
    val_targets = {
            'func': valid_func,
            'test_no_frames': args.test_no_frames}
    topn_val_param = {
        'data_params': topn_val_data_param,
        'queue_params': None,
        'targets': val_targets,
        'num_steps': val_step_num,
        'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
        'online_agg_func': train_vie.online_agg,
        'valid_loop': {'func': valid_loop}}
    validation_params = {'topn': topn_val_param}
    params['validation_params'] = validation_params

    prev_trans.start_training(params, args)


if __name__ == "__main__":
    main()
