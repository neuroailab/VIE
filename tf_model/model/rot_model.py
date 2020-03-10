from __future__ import division, print_function, absolute_import
import os, sys
import json
import numpy as np
import tensorflow as tf
import copy
import pdb
from collections import OrderedDict

from .instance_model import color_normalize
from .resnet3D_model import get_block_sizes, Model, DEFAULT_DTYPE


class ROTModel(Model):
  def __init__(self, resnet_size, data_format=None,
               dtype=DEFAULT_DTYPE):
    """
    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      resnet_version: Integer representing which version of the ResNet network
        to use. See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.
    """

    # For bigger models, we want to use "bottleneck" layers
    if resnet_size < 50:
      bottleneck = False
      final_size = 512
    else:
      bottleneck = True
      final_size = 2048

    super(ROTModel, self).__init__(
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=None,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        time_kernel_size=7,
        first_pool_size=3,
        first_pool_stride=2,
        block_sizes=get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        final_size=final_size,
        data_format=data_format)

  def _preprocess_data(self, inputs):
    org_shape = inputs.get_shape().as_list()
    inputs = tf.reshape(
        inputs, 
        [org_shape[0], 4, org_shape[1] // 4] + org_shape[2:])
    inputs = tf.reshape(
        inputs,
        [org_shape[0] * 4, org_shape[1] // 4] + org_shape[2:])
    if self.data_format == 'channels_first':
      # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
      # This provides a large performance boost on GPU. See
      # https://www.tensorflow.org/performance/performance_guide#data_formats
      inputs = tf.transpose(inputs, [0, 4, 1, 2, 3])
    return inputs

  def _get_final_dense(self, inputs):
    inputs = tf.reshape(inputs, [-1, self.final_size])
    inputs = tf.layers.dense(inputs=inputs, units=64)
    inputs = tf.identity(inputs, 'final_dense_1')
    inputs = tf.layers.dense(inputs=inputs, units=4)
    inputs = tf.identity(inputs, 'final_dense_2')

    all_logits = inputs
    all_labels = tf.tile(
        tf.expand_dims(tf.range(4, dtype=tf.int64), axis=0),
        [inputs.get_shape().as_list()[0] // 4, 1])
    all_labels = tf.reshape(all_labels, [-1])
    _, pred = tf.nn.top_k(all_logits, k=1)
    pred = tf.cast(tf.squeeze(pred), tf.int64)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(pred, all_labels), tf.float32))

    one_hot_labels = tf.one_hot(all_labels, 4)
    loss = tf.losses.softmax_cross_entropy(one_hot_labels, all_logits)
    return loss, accuracy


def build_loss_accuracy(
    inputs, train, 
    resnet_size=18,
    *args, **kwargs):
  image = color_normalize(inputs['image'])
  model = ROTModel(
      resnet_size=resnet_size)
  loss, accuracy = model(image, train, skip_final_dense=False)
  return {'loss': loss, 'accuracy': accuracy}, {}
