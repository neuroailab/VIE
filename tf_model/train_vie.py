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

from utils import tuple_get_one, online_keep_all
import utils
import config
import pdb
import data

TRN_TYPES = ['trn', 'tsrn']
SLOW_TYPES = ['slow', 'tsrn_slow', 'slowsingle_avg']
SLOWFAST_A4_TYPES = [
        'fast_a4', 'slowfast_a4', 
        'vanilla3D', 'tsrn_slowfast_a4',
        'slowfastsingle_avg']
RESNET3D_TYPES = ['3dresnet']


def get_config():
    cfg = config.Config()
    cfg.add('exp_id', type=str, required=True,
            help='Name of experiment ID')
    cfg.add('batch_size', type=int, default=128,
            help='Training batch size')
    cfg.add('test_batch_size', type=int, default=16,
            help='Testing batch size')
    cfg.add('test_no_frames', type=int, default=25,
            help='Number of frames in one video during validation')
    cfg.add('init_lr', type=float, default=0.03,
            help='Initial learning rate')
    cfg.add('target_lr', type=float, default=None,
            help='Target leraning rate for ramping up')
    cfg.add('ramp_up_epoch', type=int, default=1,
            help='Number of epoch for ramping up')
    cfg.add('gpu', type=str, required=True,
            help='Value for CUDA_VISIBLE_DEVICES')
    cfg.add('gpu_offset', type=int, default=0,
            help='GPU offset, useful for KMeans')
    cfg.add('image_dir', type=str, required=True,
            help='Directory containing dataset')
    cfg.add('val_image_dir', type=str, required=True,
            help='Directory containing dataset')
    cfg.add('data_len', type=int, default=utils.DATA_LEN_KINETICS_400,
            help='Total number of images in the input dataset')
    cfg.add('kNN_val', type=int, default=50,
            help='Not none, will use this number to do kNN validation')
    cfg.add('dataset', type=str, default=None,
            help='The name of the dataset')
    cfg.add('rotnet', type=bool, default=False,
            help='Whether finetuing 3D RotNet')
    cfg.add('train_len', type=int, default=utils.DATA_LEN_KINETICS_400,
            help='The length of the training data')
    cfg.add('val_len', type=int, default=utils.VAL_DATA_LEN_KINETICS_400,
            help='The length of the validation data')
    cfg.add('num_classes', type=int, default=400,
            help='Number of classes')    
    cfg.add('metafile_root', type=str, default=None,
            help='Directory containing metafiles')
    cfg.add('train_num_workers', type=int, default=12,
            help='Training worker number')
    cfg.add('val_num_workers', type=int, default=12,
            help='Validation worker number')
    cfg.add('train_num_steps', type=int, default=1e9,
            help='Number of training steps')

    # Training parameters
    cfg.add('weight_decay', type=float, default=1e-4,
            help='Weight decay')
    cfg.add('instance_t', type=float, default=0.07,
            help='Temperature in softmax.')
    cfg.add('instance_k', type=int, default=4096,
            help='Background neighbors to sample.')
    cfg.add('lr_boundaries', type=str, default=None,
            help='Learning rate boundaries for 10x drops')
    cfg.add('task', type=str, default='LA',
            help='Learning tasks, LA or IR')
    cfg.add('train_prep', type=str, default=None,
            help='Train preprocessing')

    cfg.add('resnet_size', type=int, default=18,
            help='ResNet size')
    cfg.add('model_type', type=str, default='single_frame',
            help='Single frame, slow, fast, or slowfast')
    cfg.add('trn_use_mean', type=bool,
            help='Using reduce_mean in TRNModel')
    cfg.add('clstr_update_fre', type=int, default=None,
            help='Update frequency for cluster')
    cfg.add('kmeans_k', type=str, default='10000',
            help='K for Kmeans')
    cfg.add('bin_interval', type=int, default=None,
            help='Bin interval for binned video dataset')
    cfg.add('part_vd', type=float, default=None,
            help='Portion of videos to use during training')
    cfg.add('trn_num_frames', type=int, default=8,
            help='Number of frames in trn style')
    cfg.add('HMDB_sample', type=bool, default=False,
            help='Whether to use HMDB sampling strategy')

    # Saving parameters
    cfg.add('port', type=int, required=True,
            help='Port number for mongodb')
    cfg.add('host', type=str, default='localhost',
            help='Host for mongodb')
    cfg.add('db_name', type=str, required=True,
            help='Name of database')
    cfg.add('col_name', type=str, required=True,
            help='Name of collection')
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

    # Loading parameters
    cfg.add('load_exp', type=str, default=None,
            help='The experiment to load from, in the format '
                 '[dbname]/[collname]/[exp_id]')
    cfg.add('load_port', type=int,
            help='Port number of mongodb for loading (defaults to saving port')
    cfg.add('load_step', type=int,
            help='Step number for loading')
    cfg.add('resume', type=bool,
            help='Flag for loading from last step of this exp_id, will override'
            ' all other loading options.')

    cfg.add('tfutils', type=bool,
            help='Whether using tfutils')
    return cfg


def loss_func(output, *args, **kwargs):
    return output['loss']


def reg_loss_in_tfutils(loss, which_device, weight_decay):
    from tfutils.multi_gpu.easy_variable_mgr import COPY_NAME_SCOPE
    curr_scope_name = '%s%i' % (COPY_NAME_SCOPE, which_device)
    # Add weight decay to the loss.
    def exclude_batch_norm_and_other_device(name):
        return 'batch_normalization' not in name and curr_scope_name in name
    l2_loss = weight_decay * tf.add_n(
            [tf.nn.l2_loss(tf.cast(v, tf.float32))
                for v in tf.trainable_variables()
                if exclude_batch_norm_and_other_device(v.name)])
    loss_all = tf.add(loss, l2_loss)
    return loss_all


def reg_loss(loss, weight_decay):
    # Add weight decay to the loss.
    def exclude_batch_norm(name):
        return 'batch_normalization' not in name
    l2_loss = weight_decay * tf.add_n(
            [tf.nn.l2_loss(tf.cast(v, tf.float32))
                for v in tf.trainable_variables()
                if exclude_batch_norm(v.name)])
    loss_all = tf.add(loss, l2_loss)
    return loss_all


def rep_loss_func(
        inputs,
        output,
        gpu_offset=0,
        **kwargs
        ):
    data_indx = output['data_indx']
    new_data_memory = output['new_data_memory']
    loss_pure = output['loss']

    memory_bank_list = output['memory_bank']
    all_labels_list = output['all_labels']
    if isinstance(memory_bank_list, tf.Variable):
        memory_bank_list = [memory_bank_list]
        all_labels_list = [all_labels_list]

    devices = ['/gpu:%i' \
               % (idx + gpu_offset) for idx in range(len(memory_bank_list))]
    update_ops = []
    for device, memory_bank, all_labels \
            in zip(devices, memory_bank_list, all_labels_list):
        with tf.device(device):
            mb_update_op = tf.scatter_update(
                    memory_bank, data_indx, new_data_memory)
            update_ops.append(mb_update_op)
            lb_update_op = tf.scatter_update(
                    all_labels, data_indx,
                    inputs['label'])
            update_ops.append(lb_update_op)

    with tf.control_dependencies(update_ops):
        # Force the updates to happen before the next batch.
        loss_pure = tf.identity(loss_pure)

    ret_dict = {'loss_pure': loss_pure}
    for key, value in output.items():
        if key.startswith('loss_'):
            ret_dict[key] = value
    return ret_dict


def online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(np.mean(v))
        # agg_res[k].append(v)
    return agg_res


def valid_perf_func_kNN(
        inputs, output, 
        instance_t,
        k, test_no_frames,
        num_classes=1000):
    curr_dist, all_labels = output
    all_labels = tuple_get_one(all_labels)
    top_dist, top_indices = tf.nn.top_k(curr_dist, k=k)
    top_labels = tf.gather(all_labels, top_indices)
    top_labels_one_hot = tf.one_hot(top_labels, num_classes)
    top_prob = tf.exp(top_dist / instance_t)
    top_labels_one_hot *= tf.expand_dims(top_prob, axis=-1)
    top_labels_one_hot = tf.reduce_mean(top_labels_one_hot, axis=1)
    top_labels_one_hot = tf.reshape(
            top_labels_one_hot,
            [-1, test_no_frames, num_classes])
    top_labels_one_hot = tf.reduce_mean(top_labels_one_hot, axis=1)
    _, curr_pred = tf.nn.top_k(top_labels_one_hot, k=1)
    curr_pred = tf.squeeze(tf.cast(curr_pred, tf.int64), axis=1)
    imagenet_top1 = tf.reduce_mean(
            tf.cast(
                tf.equal(curr_pred, inputs['label']),
                tf.float32))
    return {'top1_{k}NN'.format(k=k): imagenet_top1}


def valid_sup_func(
        inputs, output, 
        test_no_frames):
    num_classes = output.get_shape().as_list()[-1]
    curr_output = tf.nn.softmax(output)
    curr_output = tf.reshape(output, [-1, test_no_frames, num_classes])
    curr_output = tf.reduce_mean(curr_output, axis=1)

    top1_accuracy = tf.nn.in_top_k(curr_output, inputs['label'], k=1)
    top5_accuracy = tf.nn.in_top_k(curr_output, inputs['label'], k=5)
    return {'top1': top1_accuracy, 'top5': top5_accuracy}


def get_model_func_params(args):
    model_params = {
        "instance_data_len": args.data_len,
        "instance_t": args.instance_t,
        "instance_k": args.instance_k,
        "resnet_size": args.resnet_size,
        "model_type": args.model_type,
        "trn_use_mean": args.trn_use_mean,
        "kmeans_k": args.kmeans_k,
        "task": args.task,
        "num_classes": args.num_classes,
    }
    return model_params


def get_lr_from_boundary_and_ramp_up(
        global_step, boundaries, 
        init_lr, target_lr, ramp_up_epoch,
        NUM_BATCHES_PER_EPOCH):
    curr_epoch  = tf.div(
            tf.cast(global_step, tf.float32), 
            tf.cast(NUM_BATCHES_PER_EPOCH, tf.float32))
    curr_phase = (tf.minimum(curr_epoch/float(ramp_up_epoch), 1))
    curr_lr = init_lr + (target_lr-init_lr) * curr_phase

    if boundaries is not None:
        boundaries = boundaries.split(',')
        boundaries = [int(each_boundary) for each_boundary in boundaries]

        all_lrs = [
                curr_lr * (0.1 ** drop_level) \
                for drop_level in range(len(boundaries) + 1)]

        curr_lr = tf.train.piecewise_constant(
                x=global_step,
                boundaries=boundaries, values=all_lrs)
    return curr_lr


def get_save_load_params_from_arg(args):
    # save_params: defining where to save the models
    args.fre_cache_filter = args.fre_cache_filter or args.fre_filter
    cache_dir = os.path.join(
            args.cache_dir, 'models',
            args.db_name, args.col_name, args.exp_id)
    save_params = {
            'host': 'localhost',
            'port': args.port,
            'dbname': args.db_name,
            'collname': args.col_name,
            'exp_id': args.exp_id,
            'do_save': True,
            'save_initial_filters': True,
            'save_metrics_freq': args.fre_metric,
            'save_valid_freq': args.fre_valid,
            'save_filters_freq': args.fre_filter,
            'cache_filters_freq': args.fre_cache_filter,
            'cache_dir': cache_dir,
            }

    # load_params: defining where to load, if needed
    load_port = args.load_port or args.port
    load_dbname = args.db_name
    load_collname = args.col_name
    load_exp_id = args.exp_id
    load_query = None

    if not args.resume:
        if args.load_exp is not None:
            load_dbname, load_collname, load_exp_id = args.load_exp.split('/')
        if args.load_step:
            load_query = {'exp_id': load_exp_id,
                          'saved_filters': True,
                          'step': args.load_step}
            print('Load query', load_query)

    load_params = {
            'host': 'localhost',
            'port': load_port,
            'dbname': load_dbname,
            'collname': load_collname,
            'exp_id': load_exp_id,
            'do_restore': True,
            'query': load_query,
            }
    return save_params, load_params


def get_loss_lr_opt_params_from_arg(args):
    # loss_params: parameters to build the loss
    loss_params = {
        'pred_targets': [],
        'agg_func': reg_loss,
        'agg_func_kwargs': {'weight_decay': args.weight_decay},
        'loss_func': loss_func,
    }

    # learning_rate_params: build the learning rate
    # For now, just stay the same
    learning_rate_params = {
            'func': get_lr_from_boundary_and_ramp_up,
            'init_lr': args.init_lr,
            'target_lr': args.target_lr or args.init_lr,
            'NUM_BATCHES_PER_EPOCH': args.data_len // args.batch_size,
            'boundaries': args.lr_boundaries,
            'ramp_up_epoch': args.ramp_up_epoch,
            }

    # optimizer_params: use tfutils optimizer,
    # as mini batch is implemented there
    optimizer_params = {
            'optimizer': tf.train.MomentumOptimizer,
            'momentum': .9,
            }
    return loss_params, learning_rate_params, optimizer_params


def get_train_pt_loader_from_arg(args):
    if args.model_type == 'single_frame':
        train_data_loader = data.get_train_pt_loader(args)
    elif args.model_type in SLOW_TYPES:
        train_data_loader = data.get_train_slow_pt_loader(args)
    elif args.model_type in TRN_TYPES:
        train_data_loader = data.get_train_trn_pt_loader(args)
    elif args.model_type in SLOWFAST_A4_TYPES:
        train_data_loader = data.get_train_fast_a4_pt_loader(args)
    elif args.model_type in RESNET3D_TYPES:
        train_data_loader = data.get_train_3dresnet_pt_loader(args)
    else:
        train_data_loader = data.get_train_fast_pt_loader(args)
    return train_data_loader


def get_train_data_param_from_arg(args):
    if args.model_type == 'single_frame':
        train_data_param = {
                'func': data.get_placeholders,
                'batch_size': args.batch_size}
    elif args.model_type in SLOW_TYPES:
        train_data_param = {
                'func': data.get_placeholders,
                'batch_size': args.batch_size, 
                'num_frames': 4,
                'multi_frame': True}
    elif args.model_type in TRN_TYPES:
        train_data_param = {
                'func': data.get_placeholders,
                'batch_size': args.batch_size, 
                'num_frames': args.trn_num_frames,
                'multi_frame': True}
    elif args.model_type in SLOWFAST_A4_TYPES:
        train_data_param = {
                'func': data.get_placeholders,
                'batch_size': args.batch_size, 
                'num_frames': 16,
                'multi_frame': True}
    elif args.model_type in RESNET3D_TYPES:
        train_data_param = {
                'func': data.get_placeholders,
                'batch_size': args.batch_size, 
                'num_frames': 16,
                'crop_size': 112,
                'multi_frame': True}
    else:
        train_data_param = {
                'func': data.get_placeholders,
                'batch_size': args.batch_size, 
                'num_frames': 32,
                'multi_frame': True}
    return train_data_param


def get_topn_val_data_param_from_arg(args):
    if args.model_type == 'single_frame':
        topn_val_data_param = {
                'func': data.get_placeholders,
                'batch_size': args.test_batch_size,
                'num_frames': args.test_no_frames,
                'name_prefix': 'VAL'}
    elif args.model_type in SLOW_TYPES:
        topn_val_data_param = {
                'func': data.get_placeholders,
                'batch_size': args.test_batch_size,
                'num_frames': 4 * args.test_no_frames,
                'multi_frame': True,
                'multi_group': args.test_no_frames,
                'name_prefix': 'VAL'}
    elif args.model_type in TRN_TYPES:
        topn_val_data_param = {
                'func': data.get_placeholders,
                'batch_size': args.test_batch_size,
                'num_frames': args.trn_num_frames * args.test_no_frames,
                'multi_frame': True,
                'multi_group': args.test_no_frames,
                'name_prefix': 'VAL'}
    elif args.model_type in SLOWFAST_A4_TYPES:
        topn_val_data_param = {
                'func': data.get_placeholders,
                'batch_size': args.test_batch_size,
                'num_frames': 16 * args.test_no_frames,
                'multi_frame': True,
                'multi_group': args.test_no_frames,
                'name_prefix': 'VAL'}
    elif args.model_type in RESNET3D_TYPES:
        topn_val_data_param = {
                'func': data.get_placeholders,
                'batch_size': args.test_batch_size,
                'num_frames': 16 * args.test_no_frames,
                'crop_size': 112,
                'multi_frame': True,
                'multi_group': args.test_no_frames,
                'name_prefix': 'VAL'}
    else:
        topn_val_data_param = {
                'func': data.get_placeholders,
                'batch_size': args.test_batch_size,
                'num_frames': 32 * args.test_no_frames,
                'multi_frame': True,
                'multi_group': args.test_no_frames,
                'name_prefix': 'VAL'}
    return topn_val_data_param


def get_valid_loop_from_arg(args):
    val_len = args.val_len
    val_step_num = int(val_len/args.test_batch_size)
    if args.model_type == 'single_frame':
        val_data_loader = data.get_val_pt_loader(args)
    elif args.model_type in SLOW_TYPES:
        val_data_loader = data.get_val_slow_pt_loader(args)
    elif args.model_type in TRN_TYPES:
        val_data_loader = data.get_val_trn_pt_loader(args)
    elif args.model_type in SLOWFAST_A4_TYPES:
        val_data_loader = data.get_val_fast_a4_pt_loader(args)
    elif args.model_type in RESNET3D_TYPES:
        val_data_loader = data.get_val_3dresnet_pt_loader(args)
    else:
        val_data_loader = data.get_val_fast_pt_loader(args)
    val_counter = [0]
    val_data_enumerator = [enumerate(val_data_loader)]
    def valid_loop(sess, target):
        val_counter[0] += 1
        if val_counter[0] % val_step_num == 0:
            val_data_enumerator.pop()
            val_data_enumerator.append(enumerate(val_data_loader))
        _, (image, label, index) = next(val_data_enumerator[0])
        feed_dict = data.get_feeddict(image, label, index, name_prefix='VAL')
        return sess.run(target, feed_dict=feed_dict)
    return valid_loop, val_step_num


def get_params_from_arg(args):
    '''
    This function gets parameters needed for tfutils.train_from_params()
    '''

    multi_gpu = len(args.gpu.split(',')) - args.gpu_offset
    data_len = args.data_len
    NUM_BATCHES_PER_EPOCH = data_len // args.batch_size

    save_params, load_params = get_save_load_params_from_arg(args)

    # XXX: hack to set up training loop properly
    if args.kmeans_k.isdigit():
        args.kmeans_k = [int(args.kmeans_k)]
    else:
        args.kmeans_k = [int(each_k) for each_k in args.kmeans_k.split(',')]
    nn_clusterings = []
    first_step = []
    # model_params: a function that will build the model
    model_func_params = get_model_func_params(args)
    def build_output(inputs, train, **kwargs):
        res = instance_model.build_output(inputs, train, **model_func_params)
        if not train:
            return res
        outputs, logged_cfg, clustering = res
        nn_clusterings.append(clustering)
        return outputs, logged_cfg

    model_params = {'func': build_output}
    if multi_gpu > 1:
        model_params['num_gpus'] = multi_gpu
        model_params['devices'] = ['/gpu:%i' \
                                   % (idx + args.gpu_offset) \
                                   for idx in range(multi_gpu)]

    train_data_loader = get_train_pt_loader_from_arg(args)

    data_enumerator = [enumerate(train_data_loader)]
    def train_loop(sess, train_targets, num_minibatches=1, **params):
        assert num_minibatches==1, "Mini-batch not supported!"
        assert len(nn_clusterings) == multi_gpu

        global_step_vars = [v for v in tf.global_variables() \
                            if 'global_step' in v.name]
        assert len(global_step_vars) == 1
        global_step = sess.run(global_step_vars[0])

        first_flag = len(first_step) == 0
        update_fre = args.clstr_update_fre or NUM_BATCHES_PER_EPOCH
        if (global_step % update_fre == 0 or first_flag) \
                and (nn_clusterings[0] is not None):
            if first_flag:
                first_step.append(1)
            print("Recomputing clusters...")
            new_clust_labels = nn_clusterings[0].recompute_clusters(sess)
            for clustering in nn_clusterings:
                clustering.apply_clusters(sess, new_clust_labels)

        if args.part_vd is None:
            data_en_update_fre = args.data_len // args.batch_size
        else:
            new_length = int(args.data_len * args.part_vd)
            data_en_update_fre = new_length // args.batch_size

        # TODO: make this smart
        if global_step % data_en_update_fre == 0:
            data_enumerator.pop()
            data_enumerator.append(enumerate(train_data_loader))
        _, (image, label, index) = next(data_enumerator[0])
        feed_dict = data.get_feeddict(image, label, index)
        sess_res = sess.run(train_targets, feed_dict=feed_dict)
        return sess_res

    # train_params: parameters about training data
    train_data_param = get_train_data_param_from_arg(args)
    train_params = {
            'validate_first': False,
            'data_params': train_data_param,
            'queue_params': None,
            'thres_loss': float('Inf'),
            'num_steps': args.train_num_steps,
            'train_loop': {'func': train_loop},
            }

    if not args.task == 'SUP':
        ## Add other loss reports (loss_model, loss_noise)
        train_params['targets'] = {
                'func': rep_loss_func,
                'gpu_offset': args.gpu_offset,
                }

    loss_params, learning_rate_params, optimizer_params \
            = get_loss_lr_opt_params_from_arg(args)

    # validation_params: control the validation
    topn_val_data_param = get_topn_val_data_param_from_arg(args)

    if not args.task == 'SUP':
        val_targets = {
                'func': valid_perf_func_kNN,
                'k': args.kNN_val,
                'instance_t': args.instance_t,
                'test_no_frames': args.test_no_frames}
    else:
        val_targets = {
                'func': valid_sup_func,
                'test_no_frames': args.test_no_frames}

    valid_loop, val_step_num = get_valid_loop_from_arg(args)

    topn_val_param = {
        'data_params': topn_val_data_param,
        'queue_params': None,
        'targets': val_targets,
        'num_steps': val_step_num,
        'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
        'online_agg_func': online_agg,
        'valid_loop': {'func': valid_loop}
        }
    
    validation_params = {
            'topn': topn_val_param,
            }

    # Put all parameters together
    params = {
            'save_params': save_params,
            'load_params': load_params,
            'model_params': model_params,
            'train_params': train_params,
            'loss_params': loss_params,
            'learning_rate_params': learning_rate_params,
            'optimizer_params': optimizer_params,
            'log_device_placement': False,
            'validation_params': validation_params,
            'skip_check': True,
            }
    return params


def main():
    # Parse arguments
    cfg = get_config()
    args = cfg.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Get params needed, start training
    params = get_params_from_arg(args)
    if args.tfutils:
        params['loss_params']['agg_func'] = reg_loss_in_tfutils
        cache_dir = os.path.join(
                args.cache_dir, 'models_tfutils',
                args.db_name, args.col_name, args.exp_id)
        params['save_params']['cache_dir'] = cache_dir
        from tfutils import base
        base.train_from_params(**params)
    else:
        from framework import TrainFramework
        train_framework = TrainFramework(params)
        train_framework.train()


if __name__ == "__main__":
    main()
