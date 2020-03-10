from __future__ import division, print_function, absolute_import
import os, sys
import json
import numpy as np
import tensorflow as tf
import copy
import pdb
from collections import OrderedDict

from .instance_model import color_normalize
from .resnet_model import get_block_sizes, Model, DEFAULT_VERSION, DEFAULT_DTYPE

ALL_ORDERS = [
    (0,1,2,3),
    (0,2,1,3),
    (0,2,3,1),
    (0,1,3,2),
    (0,3,1,2),
    (0,3,2,1),
    (1,0,2,3),
    (1,0,3,2),
    (1,2,0,3),
    (2,0,1,3),
    (2,0,3,1),
    (2,1,0,3),
    ]


class OPNModel(Model):
  def __init__(self, resnet_size, data_format=None,
               resnet_version=DEFAULT_VERSION,
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

    super(OPNModel, self).__init__(
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=None,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        block_sizes=get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        final_size=final_size,
        resnet_version=resnet_version,
        data_format=data_format,
        dtype=dtype
    )

  def _preprocess_data(self, inputs):
    if self.data_format == 'channels_first':
      # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
      # This provides a large performance boost on GPU. See
      # https://www.tensorflow.org/performance/performance_guide#data_formats
      inputs = tf.transpose(inputs, [0, 1, 4, 2, 3])
    curr_shape = inputs.get_shape().as_list()
    self.num_frames = curr_shape[1]
    inputs = tf.reshape(inputs, [-1] + curr_shape[2:])
    return inputs

  def _build_pairwise_features(self, each_frame_out, out_dim=512):
    pairwise_features = {}
    for first_frame in range(self.num_frames):
      for second_frame in range(self.num_frames):
        if first_frame == second_frame:
          continue
        curr_pair = (first_frame, second_frame)
        curr_input_to_mlp = tf.concat(
            [each_frame_out[first_frame], each_frame_out[second_frame]], 
            axis=-1)
        curr_output = tf.layers.dense(
            inputs=curr_input_to_mlp, units=out_dim, 
            activation=tf.nn.relu, name='opn_pairwise_mlp')
        pairwise_features[curr_pair] = curr_output
    self.pairwise_features = pairwise_features

  def _build_final_mlp_for_order(self, curr_order):
    input_to_final_mlp = []
    for first_frame in range(self.num_frames):
      for second_frame in range(first_frame+1, self.num_frames):
        input_to_final_mlp.append(
            self.pairwise_features[
              (curr_order[first_frame], curr_order[second_frame])])
    input_to_final_mlp = tf.concat(input_to_final_mlp, axis=-1)
    final_mlp_output = tf.layers.dense(
        inputs=input_to_final_mlp, units=12, 
        name='opn_final_mlp')
    return final_mlp_output
  
  def _get_final_dense(self, inputs):
    inputs = tf.reshape(inputs, [-1, self.num_frames, self.final_size])
    bs = inputs.get_shape().as_list()[0]
    each_frame_out = tf.unstack(inputs, axis=1)
    self._build_pairwise_features(each_frame_out)
    
    all_logits = []
    all_labels = []
    for curr_lbl, curr_order in enumerate(ALL_ORDERS):
      _final_mlp_output = self._build_final_mlp_for_order(curr_order)
      _final_mlp_output_rev = self._build_final_mlp_for_order(
          tuple(reversed(curr_order)))
      all_logits.append(
          tf.concat([_final_mlp_output, _final_mlp_output_rev], axis=0))
      all_labels.append(tf.ones((bs * 2), dtype=tf.int64) * curr_lbl)
    all_logits = tf.concat(all_logits, axis=0)
    all_labels = tf.concat(all_labels, axis=0)

    _, pred = tf.nn.top_k(all_logits, k=1)
    pred = tf.cast(tf.squeeze(pred), tf.int64)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(pred, all_labels), tf.float32))

    one_hot_labels = tf.one_hot(all_labels, 12)
    loss = tf.losses.softmax_cross_entropy(one_hot_labels, all_logits)
    return loss, accuracy


def build_loss_accuracy(
    inputs, train, 
    resnet_size=18,
    *args, **kwargs):
  image = color_normalize(inputs['image'])
  model = OPNModel(
      resnet_size=resnet_size)
  loss, accuracy = model(image, train, skip_final_dense=False)
  return {'loss': loss, 'accuracy': accuracy}, {}
