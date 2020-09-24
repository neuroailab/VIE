from __future__ import division, print_function, absolute_import
import os, sys
import json
import numpy as np
import tensorflow as tf
import copy
import pdb
from collections import OrderedDict

from . import resnet_model
from . import resnet_model_slowfast
from . import resnet3D_model
from .memory_bank import MemoryBank
from .self_loss import get_selfloss, assert_shape, DATA_LEN_IMAGENET_FULL

MEMORY_DIM = 128


def flatten(layer_out):
    curr_shape = layer_out.get_shape().as_list()
    if len(curr_shape) > 2:
        layer_out = tf.reshape(layer_out, [curr_shape[0], -1])
    return layer_out


def get_resnet_all_layers(ending_points, get_all_layers):
    ending_dict = {}
    get_all_layers = get_all_layers.split(',')
    for idx, layer_out in enumerate(ending_points):
        # TODO: make this flexible?
        if len(layer_out.get_shape().as_list()) == 5:
            layer_out = tf.reduce_mean(layer_out, axis=2)

        if str(idx) in get_all_layers:
            feat_size = np.prod(layer_out.get_shape().as_list()[1:])
            if feat_size > 200000:
                pool_size = 2
                if feat_size / 4 > 200000:
                    pool_size = 4
                layer_out = tf.transpose(layer_out, [0, 2, 3, 1])
                layer_out = tf.nn.avg_pool(
                        layer_out,
                        ksize=[1,pool_size,pool_size,1],
                        strides=[1,pool_size,pool_size,1],
                        padding='SAME')
            layer_out = flatten(layer_out)
            ending_dict[str(idx)] = layer_out

        avg_name = '%i-avg' % idx
        if avg_name in get_all_layers:
            layer_out = tf.reduce_mean(layer_out, axis=[2,3])
            ending_dict[avg_name] = layer_out

        time_avg_name = '%i-time-avg' % idx
        if time_avg_name in get_all_layers:
            num_frames = 4
            bs_nof = layer_out.get_shape().as_list()[0]
            layer_out = tf.reduce_mean(
                    tf.reshape(
                        layer_out, 
                        [bs_nof // num_frames, num_frames, -1]), 
                    axis=1)
            ending_dict[time_avg_name] = layer_out

        sftime_avg_name = '%i-sftime-avg' % idx
        if sftime_avg_name in get_all_layers:
            num_frames = 16
            bs_nof = layer_out.get_shape().as_list()[0]
            layer_out = tf.reduce_mean(
                    tf.reshape(
                        layer_out, 
                        [bs_nof // num_frames, num_frames, -1]), 
                    axis=1)
            ending_dict[sftime_avg_name] = layer_out
    return ending_dict


def get_combined_resnet_all_layers(combined_ending_points, *args, **kwargs):
    final_ending_dict = None
    for now_ending_points in combined_ending_points:
        now_ending_dict = get_resnet_all_layers(
                now_ending_points, *args, **kwargs)
        if final_ending_dict is None:
            final_ending_dict = {}
            for key in now_ending_dict:
                final_ending_dict[key] = []
        for key, out in now_ending_dict.items():
            assert key in final_ending_dict
            final_ending_dict[key].append(out)

    concat_ending_dict = {}
    for key, out_list in final_ending_dict.items():
        concat_ending_dict[key] = tf.concat(out_list, axis=-1)
    return concat_ending_dict


def color_normalize(image):
    image = tf.cast(image, tf.float32)
    image = tf.div(image, tf.constant(255, dtype=tf.float32))

    image_shape = image.get_shape().as_list()
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    mean = tf.reshape(mean, [1] * (len(image_shape) - 1) + [3])
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    std = tf.reshape(std, [1] * (len(image_shape) - 1) + [3])

    image = (image - mean) / std
    return image


def resnet_embedding(image,
                     data_format=None, train=False,
                     resnet_size=18,
                     model_type='single_frame',
                     trn_use_mean=False,
                     get_all_layers=None,
                     skip_final_dense=False,
                     num_classes=MEMORY_DIM,
                     final_pooling=None,
                     slowfast_single_pooling=False):
    image = color_normalize(image)

    model_kwargs = {}
    model_class = resnet_model.SingleFrameModel
    if model_type == 'slow': 
        model_class = resnet_model_slowfast.SlowModel
    elif model_type == 'trn': 
        model_class = resnet_model.TRNModel
        model_kwargs = {'use_mean': trn_use_mean}
    elif model_type in ['tsrn', 'tsrn_slow', 'tsrn_slowfast_a4']: 
        model_class = resnet_model.TSRNModel
    elif model_type in ['fast', 'fast_a4']: 
        model_class = resnet_model_slowfast.FastModel
    elif model_type == 'slowfast_a4': 
        model_class = resnet_model_slowfast.SlowFastModel
    elif model_type in ['vanilla3D', '3dresnet']:
        model_class = resnet3D_model.Vanilla3DModel
    elif model_type == 'trn_f4_tile':
        model_class = resnet_model.TRNTileModel
        model_kwargs = {
                'trn_num_frames': 4,
                'use_mean': trn_use_mean}
    elif model_type == 'slowsingle':
        model_class = resnet_model_slowfast.SlowSingleModel
    elif model_type == 'slowsingle_avg':
        model_class = resnet_model_slowfast.SlowSingleModel
        model_kwargs = {
                'multi_frame_choice': 'avg'}
    elif model_type == 'slowfastsingle':
        model_class = resnet_model_slowfast.SlowFastSingleModel
    elif model_type == 'slowfastsingle_avg':
        model_class = resnet_model_slowfast.SlowFastSingleModel
        model_kwargs = {
                'multi_frame_choice': 'avg'}
    model = model_class(
        resnet_size=resnet_size, data_format=data_format,
        num_classes=num_classes, **model_kwargs)

    if skip_final_dense and get_all_layers is None:
        return model(image, training=train, skip_final_dense=True, final_pooling=final_pooling)
        
    if get_all_layers:
        if model_type == 'slowfastsingle_avg' and slowfast_single_pooling:
            _, ending_points = model(
                image, True, training=train, get_all_layers=get_all_layers)
        else:
            _, ending_points = model(
                image, training=train, get_all_layers=get_all_layers)
        if model_type in ['slowsingle', 'slowsingle_avg', 
                          'slowfastsingle', 'slowfastsingle_avg']:
            all_layers = get_combined_resnet_all_layers(
                    ending_points, get_all_layers)
        else:
            all_layers = get_resnet_all_layers(ending_points, get_all_layers)
        return all_layers

    model_out = model(image, training=train, skip_final_dense=False)
    return model_out


def repeat_1d_tensor(t, num_reps):
    ret = tf.tile(tf.expand_dims(t, axis=1), (1, num_reps))
    return ret


class InstanceModel(object):
    def __init__(self,
                 inputs, output,
                 memory_bank,
                 instance_k=4096,
                 instance_t=0.07,
                 instance_m=0.5,
                 **kwargs):
        self.inputs = inputs
        self.embed_output = output
        self.batch_size, self.out_dim = self.embed_output.get_shape().as_list()
        self.memory_bank = memory_bank

        self.instance_data_len = memory_bank.size
        self.instance_k = instance_k
        self.instance_t = instance_t
        self.instance_m = instance_m

    def _softmax(self, dot_prods):
        instance_Z = tf.constant(
            2876934.2 / 1281167 * self.instance_data_len,
            dtype=tf.float32)
        return tf.exp(dot_prods / self.instance_t) / instance_Z

    def updated_new_data_memory(self):
        data_indx = self.inputs['index'] # [bs]
        data_memory = self.memory_bank.at_idxs(data_indx)
        new_data_memory = (data_memory * self.instance_m
                           + (1 - self.instance_m) * self.embed_output)
        return tf.nn.l2_normalize(new_data_memory, axis=1)

    def __get_lbl_equal(self, each_k_idx, cluster_labels, top_idxs, k):
        batch_labels = tf.gather(
                cluster_labels[each_k_idx], 
                self.inputs['index'])
        if k > 0:
            top_cluster_labels = tf.gather(cluster_labels[each_k_idx], top_idxs)
            batch_labels = repeat_1d_tensor(batch_labels, k)
            curr_equal = tf.equal(batch_labels, top_cluster_labels)
        else:
            curr_equal = tf.equal(
                    tf.expand_dims(batch_labels, axis=1), 
                    tf.expand_dims(cluster_labels[each_k_idx], axis=0))
        return curr_equal

    def __get_prob_from_equal(self, curr_equal, exponents):
        probs = tf.reduce_sum(
            tf.where(
                curr_equal,
                x=exponents, y=tf.zeros_like(exponents),
            ), axis=1)
        probs /= tf.reduce_sum(exponents, axis=1)
        return probs

    def get_cluster_classification_loss(
            self, cluster_labels, 
            k=None):
        if not k:
            k = self.instance_k
        # ignore all but the top k nearest examples
        all_dps = self.memory_bank.get_all_dot_products(self.embed_output)
        top_dps, top_idxs = tf.nn.top_k(all_dps, k=k, sorted=False)
        if k > 0:
            exponents = self._softmax(top_dps)
        else:
            exponents = self._softmax(all_dps)

        no_kmeans = cluster_labels.get_shape().as_list()[0]
        all_equal = None
        for each_k_idx in range(no_kmeans):
            curr_equal = self.__get_lbl_equal(
                    each_k_idx, cluster_labels, top_idxs, k)

            if all_equal is None:
                all_equal = curr_equal
            else:
                all_equal = tf.logical_or(all_equal, curr_equal)
        probs = self.__get_prob_from_equal(all_equal, exponents)

        assert_shape(probs, [self.batch_size])
        loss = -tf.reduce_mean(tf.log(probs + 1e-7))
        return loss, self.inputs['index']

    def compute_data_prob(self, selfloss):
        data_indx = self.inputs['index']
        logits = selfloss.get_closeness(data_indx, self.embed_output)
        return self._softmax(logits)

    def compute_noise_prob(self):
        noise_indx = tf.random_uniform(
            shape=(self.batch_size, self.instance_k),
            minval=0,
            maxval=self.instance_data_len,
            dtype=tf.int64)
        noise_probs = self._softmax(
            self.memory_bank.get_dot_products(self.embed_output, noise_indx))
        return noise_probs

    def get_losses(self, data_prob, noise_prob):
        assert_shape(data_prob, [self.batch_size])
        assert_shape(noise_prob, [self.batch_size, self.instance_k])

        base_prob = 1.0 / self.instance_data_len
        eps = 1e-7
        ## Pmt
        data_div = data_prob + (self.instance_k*base_prob + eps)
        ln_data = tf.log(data_prob / data_div)
        ## Pon
        noise_div = noise_prob + (self.instance_k*base_prob + eps)
        ln_noise = tf.log((self.instance_k*base_prob) / noise_div)

        curr_loss = -(tf.reduce_sum(ln_data) \
                      + tf.reduce_sum(ln_noise)) / self.batch_size
        return curr_loss, \
            -tf.reduce_sum(ln_data)/self.batch_size, \
            -tf.reduce_sum(ln_noise)/self.batch_size


def build_output(
        inputs, train, 
        resnet_size=18,
        model_type='single_frame',
        trn_use_mean=False,
        kmeans_k=[10000],
        task='LA',
        num_classes=400,
        **kwargs):
    # This will be stored in the db
    logged_cfg = {'kwargs': kwargs}

    embedding_kwargs = {
            'train': train,
            'resnet_size': resnet_size,
            'model_type': model_type,
            'trn_use_mean': trn_use_mean}
    if task == 'SUP':
        output = resnet_embedding(
                inputs['image'], num_classes=num_classes,
                **embedding_kwargs)

        if not train:
            return output, logged_cfg

        one_hot_labels = tf.one_hot(inputs['label'], num_classes)
        loss = tf.losses.softmax_cross_entropy(one_hot_labels, output)
        return {'loss': loss}, logged_cfg, None

    data_len = kwargs.get('instance_data_len', DATA_LEN_IMAGENET_FULL)
    with tf.variable_scope('instance', reuse=tf.AUTO_REUSE):
        all_labels = tf.get_variable(
            'all_labels',
            initializer=tf.zeros_initializer,
            shape=(data_len,),
            trainable=False,
            dtype=tf.int64,
        )
        memory_bank = MemoryBank(data_len, MEMORY_DIM)

        if task == 'LA':
            lbl_init_values = tf.range(data_len, dtype=tf.int64)
            no_kmeans_k = len(kmeans_k)
            lbl_init_values = tf.tile(
                    tf.expand_dims(lbl_init_values, axis=0),
                    [no_kmeans_k, 1])
            cluster_labels = tf.get_variable(
                'cluster_labels',
                initializer=lbl_init_values,
                trainable=False, dtype=tf.int64,
            )

    output = resnet_embedding(
            inputs['image'], **embedding_kwargs)
    output = tf.nn.l2_normalize(output, axis=1)

    if not train:
        all_dist = memory_bank.get_all_dot_products(output)
        return [all_dist, all_labels], logged_cfg
    model_class = InstanceModel(
        inputs=inputs, output=output,
        memory_bank=memory_bank,
        **kwargs)
    nn_clustering = None
    other_losses = {}
    if task == 'LA':
        from .cluster_km import Kmeans
        nn_clustering = Kmeans(kmeans_k, memory_bank, cluster_labels)
        loss, new_nns = model_class.get_cluster_classification_loss(
                cluster_labels)
    else:
        selfloss = get_selfloss(memory_bank, **kwargs)
        data_prob = model_class.compute_data_prob(selfloss)
        noise_prob = model_class.compute_noise_prob()
        losses = model_class.get_losses(data_prob, noise_prob)
        loss, loss_model, loss_noise = losses
        other_losses['loss_model'] = loss_model
        other_losses['loss_noise'] = loss_noise

    new_data_memory = model_class.updated_new_data_memory()
    ret_dict = {
        "loss": loss,
        "data_indx": inputs['index'],
        "memory_bank": memory_bank.as_tensor(),
        "new_data_memory": new_data_memory,
        "all_labels": all_labels,
    }
    ret_dict.update(other_losses)
    return ret_dict, logged_cfg, nn_clustering


def build_transfer_output(
        inputs, train, 
        resnet_size=18,
        model_type='single_frame',
        get_all_layers=None, 
        num_classes=1000,
        **kwargs):
    # This will be stored in the db
    logged_cfg = {'kwargs': kwargs}
    input_image = inputs['image']
    num_crop = None
    curr_image_shape = input_image.get_shape().as_list()
    batch_size = curr_image_shape[0]
    '''
    if len(curr_image_shape) > 4:
        num_crop = curr_image_shape[1]
        input_image = tf.reshape(input_image, [-1] + curr_image_shape[2:])
    '''
    resnet_output = resnet_embedding(
        input_image,
        train=False,
        resnet_size=resnet_size,
        model_type=model_type,
        skip_final_dense=True,
        get_all_layers=get_all_layers)

    with tf.variable_scope('instance', reuse=tf.AUTO_REUSE):
        init_builder = tf.contrib.layers.variance_scaling_initializer()
        if not get_all_layers:
            class_output = tf.layers.dense(
                inputs=resnet_output, units=num_classes,
                kernel_initializer=init_builder,
                trainable=True,
                name='transfer_dense')
        else:
            class_output = OrderedDict()
            for key, curr_out in resnet_output.items():
                class_output[key] = tf.layers.dense(
                    inputs=curr_out, units=num_classes,
                    kernel_initializer=init_builder,
                    trainable=True,
                    name='transfer_dense_{name}'.format(name=key))

    def __get_loss_accuracy(curr_output):
        if num_crop:
            curr_output = tf.nn.softmax(curr_output)
            curr_output = tf.reshape(curr_output, [batch_size, num_crop, -1])
            curr_output = tf.reduce_mean(curr_output, axis=1)
        _, pred = tf.nn.top_k(curr_output, k=1)
        pred = tf.cast(tf.squeeze(pred), tf.int64)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(pred, inputs['label']), tf.float32)
        )

        one_hot_labels = tf.one_hot(inputs['label'], num_classes)
        loss = tf.losses.softmax_cross_entropy(one_hot_labels, curr_output)
        return loss, accuracy
    if not get_all_layers:
        loss, accuracy = __get_loss_accuracy(class_output)
    else:
        loss = []
        accuracy = OrderedDict()
        for key, curr_out in class_output.items():
            curr_loss, curr_acc = __get_loss_accuracy(curr_out)
            loss.append(curr_loss)
            accuracy[key] = curr_acc
        loss = tf.reduce_sum(loss)

    if not train:
        return accuracy, logged_cfg
    return [loss, accuracy], logged_cfg


def build_KN_transfer_output(
        inputs, train, 
        finetune_conv=False,
        resnet_size=18,
        model_type='single_frame',
        get_all_layers=None, 
        num_classes=400,
        dropout=None,
        final_pooling=None,
        slowfast_single_pooling=False,
        **kwargs):
    # This will be stored in the db
    logged_cfg = {'kwargs': kwargs}
    input_image = inputs['image']

    bn_train = train and finetune_conv
    resnet_output = resnet_embedding(
        input_image,
        train=bn_train,
        resnet_size=resnet_size,
        model_type=model_type,
        skip_final_dense=True,
        get_all_layers=get_all_layers,
        final_pooling=final_pooling,
        slowfast_single_pooling=slowfast_single_pooling)

    with tf.variable_scope('instance', reuse=tf.AUTO_REUSE):
        init_builder = tf.contrib.layers.variance_scaling_initializer()
        if not get_all_layers:
            if dropout is not None and train:
                resnet_output = tf.layers.dropout(inputs=resnet_output, rate=dropout)
            class_output = tf.layers.dense(
                inputs=resnet_output, units=num_classes,
                kernel_initializer=init_builder,
                trainable=True,
                name='transfer_dense')
        else:
            class_output = OrderedDict()
            for key, curr_out in resnet_output.items():
                if dropout is not None and train:
                    curr_out = tf.layers.dropout(inputs=curr_out, rate=dropout)
                class_output[key] = tf.layers.dense(
                    inputs=curr_out, units=num_classes,
                    kernel_initializer=init_builder,
                    trainable=True,
                    name='transfer_dense_{name}'.format(name=key))
    if not train:
        return class_output, logged_cfg

    def __get_loss_accuracy(curr_output):
        _, pred = tf.nn.top_k(curr_output, k=1)
        pred = tf.cast(tf.squeeze(pred), tf.int64)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(pred, inputs['label']), tf.float32)
        )

        one_hot_labels = tf.one_hot(inputs['label'], num_classes)
        loss = tf.losses.softmax_cross_entropy(one_hot_labels, curr_output)
        return loss, accuracy

    if not get_all_layers:
        loss, accuracy = __get_loss_accuracy(class_output)
    else:
        loss = []
        accuracy = []
        for key, curr_out in class_output.items():
            curr_loss, curr_acc = __get_loss_accuracy(curr_out)
            loss.append(curr_loss)
            accuracy.append(curr_acc)
        loss = tf.reduce_sum(loss)
        accuracy = tf.reduce_mean(accuracy)

    return {'loss': loss, 'accuracy': accuracy}, logged_cfg
