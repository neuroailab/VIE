"""With 3D convolutions in layers described by SlowFast paper
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import tensorflow as tf
from .resnet_model import batch_norm, fixed_padding, \
        conv2d_fixed_padding, get_block_sizes
from .resnet_model import SingleFrameModel
_NUM_CLASSES = 128
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES

ENDING_POINTS = []


def fixed_padding_3d(inputs, kernel_size, time_kernel_size, data_format):
  """Pads the input along the spatial and time dimensions 
     independently of input size.
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  time_pad_total = time_kernel_size - 1
  time_pad_beg = time_pad_total // 2
  time_pad_end = time_pad_total - time_pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], 
                                    [time_pad_beg, time_pad_end],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], 
                                    [time_pad_beg, time_pad_end],
                                    [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv3d_fixed_padding(inputs, filters, kernel_size, time_kernel_size, 
                         strides, data_format, time_stride=1):
  """Strided 3-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1 or time_stride > 1:
    inputs = fixed_padding_3d(inputs, kernel_size, 
                              time_kernel_size, data_format)

  return tf.layers.conv3d(
      inputs=inputs, filters=filters, 
      kernel_size=(time_kernel_size, kernel_size, kernel_size), 
      strides=(time_stride, strides, strides), 
      padding=('SAME' if (strides == 1 and time_stride==1) else 'VALID'),
      use_bias=False, kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def _building_block_v2(inputs, filters, training, 
                       projection_shortcut, strides,
                       data_format, time_kernel_size):
  """A single block for ResNet v2, without a bottleneck.

  Batch normalization then ReLu then convolution as described by:
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/pdf/1603.05027.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)
  ENDING_POINTS.append(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv3d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, 
      time_kernel_size=time_kernel_size, strides=strides,
      data_format=data_format)

  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)
  inputs = conv3d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, 
      time_kernel_size=1, strides=1,
      data_format=data_format)

  return inputs + shortcut


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name, data_format, time_kernel_size):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    bottleneck: Is the block created a bottleneck block.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block layer.
  """

  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = filters * 4 if bottleneck else filters

  def projection_shortcut(inputs):
    return conv3d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, 
        time_kernel_size=1, strides=strides,
        data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                    data_format, time_kernel_size)

  for which_blck in range(1, blocks):
    inputs = block_fn(inputs, filters, training, 
                      None, 1, data_format, time_kernel_size)

  return tf.identity(inputs, name)


class Model(object):
  """Base class for building the Resnet Model."""

  def __init__(self, resnet_size, bottleneck, num_classes, 
               num_filters, kernel_size, conv_stride, time_kernel_size,
               first_pool_size, first_pool_stride,
               block_sizes, block_strides, block_time_kernel_size,
               final_size, data_format=None,
               model_name_scope='resnet_model'):
    """Creates a model for classifying an image. Use V2.

    Args:
      resnet_size: A single integer for the size of the ResNet model.
      bottleneck: Use regular blocks or bottleneck blocks.
      num_classes: The number of classes used as labels.
      num_filters: The number of filters to use for the first block layer
        of the model. This number is then doubled for each subsequent block
        layer.
      kernel_size: The kernel size to use for convolution.
      conv_stride: stride size for the initial convolutional layer
      first_pool_size: Pool size to be used for the first pooling layer.
        If none, the first pooling layer is skipped.
      first_pool_stride: stride size for the first pooling layer. Not used
        if first_pool_size is None.
      block_sizes: A list containing n values, where n is the number of sets of
        block layers desired. Each value should be the number of blocks in the
        i-th set.
      block_strides: List of integers representing the desired stride size for
        each of the sets of block layers. Should be same length as block_sizes.
      final_size: The expected size of the model after the second pooling.
      data_format: Input format ('channels_last', 'channels_first', or None).
        If set to None, the format is dependent on whether a GPU is available.

    Raises:
      ValueError: if invalid version is selected.
    """
    self.resnet_size = resnet_size

    if not data_format:
      data_format = (
          'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    self.resnet_version = 2

    self.bottleneck = bottleneck
    if bottleneck:
      self.block_fn = _bottleneck_block_v2
    else:
      self.block_fn = _building_block_v2

    self.data_format = data_format
    self.num_classes = num_classes
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.conv_stride = conv_stride
    self.time_kernel_size = time_kernel_size
    self.first_pool_size = first_pool_size
    self.first_pool_stride = first_pool_stride
    self.block_sizes = block_sizes
    self.block_strides = block_strides
    self.block_time_kernel_size = block_time_kernel_size
    self.final_size = final_size
    self.dtype = tf.float32
    self.pre_activation = True
    self.model_name_scope = model_name_scope

  def _custom_dtype_getter(self, getter, name, shape=None, dtype=DEFAULT_DTYPE,
                           *args, **kwargs):
    """Creates variables in fp32, then casts to fp16 if necessary.

    This function is a custom getter. A custom getter is a function with the
    same signature as tf.get_variable, except it has an additional getter
    parameter. Custom getters can be passed as the `custom_getter` parameter of
    tf.variable_scope. Then, tf.get_variable will call the custom getter,
    instead of directly getting a variable itself. This can be used to change
    the types of variables that are retrieved with tf.get_variable.
    The `getter` parameter is the underlying variable getter, that would have
    been called if no custom getter was used. Custom getters typically get a
    variable with `getter`, then modify it in some way.

    This custom getter will create an fp32 variable. If a low precision
    (e.g. float16) variable was requested it will then cast the variable to the
    requested dtype. The reason we do not directly create variables in low
    precision dtypes is that applying small gradients to such variables may
    cause the variable not to change.

    Args:
      getter: The underlying variable getter, that has the same signature as
        tf.get_variable and returns a variable.
      name: The name of the variable to get.
      shape: The shape of the variable to get.
      dtype: The dtype of the variable to get. Note that if this is a low
        precision dtype, the variable will be created as a tf.float32 variable,
        then cast to the appropriate dtype
      *args: Additional arguments to pass unmodified to getter.
      **kwargs: Additional keyword arguments to pass unmodified to getter.

    Returns:
      A variable which is cast to fp16 if necessary.
    """

    if dtype in CASTABLE_TYPES:
      var = getter(name, shape, tf.float32, *args, **kwargs)
      return tf.cast(var, dtype=dtype, name=name + '_cast')
    else:
      return getter(name, shape, dtype, *args, **kwargs)

  def _model_variable_scope(self):
    """Returns a variable scope that the model should be created under.

    If self.dtype is a castable type, model variable will be created in fp32
    then cast to self.dtype before being used.

    Returns:
      A variable scope for the model.
    """

    return tf.variable_scope(self.model_name_scope,
                             custom_getter=self._custom_dtype_getter)

  def _add_lateral_conn(self, inputs, lateral_inputs):
    channel_dim = 1 if self.data_format == 'channels_first' else -1
    time_dim = 2 if self.data_format == 'channels_first' else 1
    channel_num_in_lateral = lateral_inputs.shape[channel_dim]
    time_num_in_lateral = lateral_inputs.shape[time_dim]
    time_num_in_inputs = inputs.shape[time_dim]
    alpha = time_num_in_lateral // time_num_in_inputs
    lateral_inputs = conv3d_fixed_padding(
        inputs=lateral_inputs, filters=channel_num_in_lateral * 2, 
        kernel_size=1, time_kernel_size=5,
        strides=1, data_format=self.data_format,
        time_stride=alpha)
    inputs = tf.concat([inputs, lateral_inputs], axis=channel_dim)
    return inputs

  def __call__(
      self, inputs, training, get_all_layers=None, skip_final_dense=False,
      return_lateral=False, add_lateral=None, 
      skip_final_dense_with_pool=False, final_pooling=None,
      slowfast_single_pooling=False):
    """Add operations to classify a batch of input images.

    Args:
      inputs: A Tensor representing a batch of input images.
      training: A boolean. Set to True to add operations required only when
        training the classifier.

    Returns:
      A logits Tensor with shape [<batch_size>, self.num_classes].
    """
    global ENDING_POINTS
    ENDING_POINTS = []

    outputs_for_lateral = {}

    with self._model_variable_scope():
      if self.data_format == 'channels_first':
        # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
        # This provides a large performance boost on GPU. See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        inputs = tf.transpose(inputs, [0, 4, 1, 2, 3])

      inputs = conv3d_fixed_padding(
          inputs=inputs, filters=self.num_filters, 
          kernel_size=self.kernel_size, time_kernel_size=self.time_kernel_size,
          strides=self.conv_stride, data_format=self.data_format)
      inputs = tf.identity(inputs, 'initial_conv')

      # We do not include batch normalization or activation functions in V2
      # for the initial conv1 because the first ResNet unit will perform these
      # for both the shortcut and non-shortcut paths as part of the first
      # block's projection. Cf. Appendix of [2].
      if self.resnet_version == 1:
        inputs = batch_norm(inputs, training, self.data_format)
        inputs = tf.nn.relu(inputs)

      if self.first_pool_size:
        inputs = tf.layers.max_pooling3d(
            inputs=inputs, 
            pool_size=[1, self.first_pool_size, self.first_pool_size],
            strides=[1, self.first_pool_stride, self.first_pool_stride], 
            padding='SAME',
            data_format=self.data_format)
        inputs = tf.identity(inputs, 'initial_max_pool')
      if return_lateral:
        outputs_for_lateral['pool1'] = inputs
      if add_lateral is not None:
        inputs = self._add_lateral_conn(inputs, add_lateral['pool1'])

      ENDING_POINTS.append(inputs)

      for i, num_blocks in enumerate(self.block_sizes):
        num_filters = self.num_filters * (2**i)
        inputs = block_layer(
            inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
            block_fn=self.block_fn, blocks=num_blocks,
            strides=self.block_strides[i], training=training,
            time_kernel_size=self.block_time_kernel_size[i],
            name='block_layer{}'.format(i + 1), data_format=self.data_format)
        curr_res_name = 'res{}'.format(i + 1)
        if return_lateral:
          outputs_for_lateral[curr_res_name] = inputs
        if add_lateral is not None:
          inputs = self._add_lateral_conn(inputs, add_lateral[curr_res_name])

      # Only apply the BN and ReLU for model that does pre_activation in each
      # building/bottleneck block, eg resnet V2.
      if self.pre_activation:
        inputs = batch_norm(inputs, training, self.data_format)
        inputs = tf.nn.relu(inputs)
        if slowfast_single_pooling:
          axes = [2, 3, 4] if self.data_format == 'channels_first' else [1, 2, 3]
          inputs = tf.reduce_mean(inputs, axes, keepdims=False)
          inputs = tf.identity(inputs, 'final_reduce_mean')
      ENDING_POINTS.append(inputs)

      if skip_final_dense:
        bs = inputs.get_shape().as_list()[0]
        inputs = tf.reduce_mean(inputs, axis=2)
        if final_pooling is None:
          inputs = tf.reshape(inputs, [bs, -1])
          if return_lateral:
            return inputs, outputs_for_lateral
          else:
            return inputs
        if final_pooling == 1:
          axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
          inputs = tf.reduce_mean(inputs, axes, keepdims=False)
          inputs = tf.reshape(inputs, [bs, -1])
          if return_lateral:
            return inputs, outputs_for_lateral
          else:
            return inputs
        if final_pooling == 2:
          inputs = tf.layers.average_pooling2d(inputs=inputs, pool_size=4, 
                                              strides=4, padding='same', 
                                              data_format=self.data_format)
          return tf.reshape(inputs, [bs, 2 * 2 * self.final_size])
        if final_pooling == 3:
          inputs = tf.layers.average_pooling2d(inputs=inputs, pool_size=3, 
                                              strides=3, padding='same', 
                                              data_format=self.data_format)
          return tf.reshape(inputs, [bs, 3 * 3 * self.final_size])


      # The current top layer has shape
      # `batch_size x pool_size x pool_size x final_size`.
      # ResNet does an Average Pooling layer over pool_size,
      # but that is the same as doing a reduce_mean. We do a reduce_mean
      # here because it performs better than AveragePooling2D.
      if not slowfast_single_pooling:
        axes = [2, 3, 4] if self.data_format == 'channels_first' else [1, 2, 3]
        inputs = tf.reduce_mean(inputs, axes, keepdims=False)
        inputs = tf.identity(inputs, 'final_reduce_mean')

      if skip_final_dense_with_pool:
        outputs_for_lateral['final'] = inputs
        if return_lateral:
          return inputs, outputs_for_lateral
        else:
          return inputs
      if add_lateral is not None:
        inputs = tf.concat([inputs, add_lateral['final']], axis=1)

      inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
      inputs = tf.identity(inputs, 'final_dense')
      if not get_all_layers:
        return inputs
      else:
        return inputs, ENDING_POINTS


class SlowModel(Model):
  """Model class with appropriate defaults for Imagenet data."""

  def __init__(
      self, resnet_size, data_format=None, num_classes=_NUM_CLASSES,
      **kwargs):
    """These are the parameters that work for Imagenet data.

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

    super(SlowModel, self).__init__(
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=num_classes,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        time_kernel_size=1,
        first_pool_size=3,
        first_pool_stride=2,
        block_sizes=get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        block_time_kernel_size=[1, 1, 3, 3],
        final_size=final_size,
        data_format=data_format, **kwargs)


class FastModel(Model):
  """Model class with appropriate defaults for Imagenet data."""

  def __init__(
      self, resnet_size, data_format=None, num_classes=_NUM_CLASSES,
      **kwargs):
    """These are the parameters that work for Imagenet data.

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
      final_size = 64
    else:
      bottleneck = True
      final_size = 256

    super(FastModel, self).__init__(
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=num_classes,
        num_filters=8,
        kernel_size=7,
        conv_stride=2,
        time_kernel_size=5,
        first_pool_size=3,
        first_pool_stride=2,
        block_sizes=get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        block_time_kernel_size=[3, 3, 3, 3],
        final_size=final_size,
        data_format=data_format, **kwargs)


class SlowFastModel(object):
  """Build SlowFast model with lateral connections"""

  def __init__(self, alpha=4, *args, **kwargs):
    self.alpha = alpha

    self.fast_model = FastModel(
        model_name_scope='resnet_model_fast', *args, **kwargs)
    self.slow_model = SlowModel(
        model_name_scope='resnet_model_slow', *args, **kwargs)

  def __call__(self, inputs, slowfast_single_pooling=False, *args, **kwargs):
    fast_output, outputs_for_lateral = self.fast_model(
        inputs, 
        return_lateral=True, skip_final_dense_with_pool=True, slowfast_single_pooling=slowfast_single_pooling, 
        *args, **kwargs)
    
    # Assume the time dimension is 1
    slow_inputs = inputs[:, ::self.alpha]
    final_output = self.slow_model(
        slow_inputs,
        add_lateral=outputs_for_lateral, slowfast_single_pooling=slowfast_single_pooling, 
        *args, **kwargs)
    if kwargs.get('skip_final_dense', False):
      final_output = tf.concat([fast_output, final_output], axis=1)
    return final_output


class SlowSingleModel(object):
  def __init__(self, multi_frame_choice='choose_one', *args, **kwargs):
    self.single_frame_model = SingleFrameModel(*args, **kwargs)
    self.slow_model = SlowModel(
        model_name_scope='resnet_model_slow', *args, **kwargs)
    self.multi_frame_choice = multi_frame_choice

  def _get_single_frame_layers(self, inputs, slowfast_single_pooling=False, *args, **kwargs):
    if self.multi_frame_choice == 'choose_one':
      _, single_frame_layers = self.single_frame_model(
          inputs[:, 0], *args, **kwargs)
    elif self.multi_frame_choice == 'avg':
      curr_input_shape = inputs.get_shape().as_list()
      num_frames = curr_input_shape[1]
      new_inputs = tf.reshape(inputs, [-1] + curr_input_shape[2:])
      _, mlt_single_frame_layers = self.single_frame_model(
          new_inputs, slowfast_single_pooling=slowfast_single_pooling, *args, **kwargs)
      single_frame_layers = []
      for each_layer in mlt_single_frame_layers:
        mean_time_layer = tf.reduce_mean(
            tf.reshape(
              each_layer, 
              [-1, num_frames] + each_layer.get_shape().as_list()[1:]),
            axis=1)
        single_frame_layers.append(mean_time_layer)
    else:
      raise NotImplementedError
    return single_frame_layers

  def __call__(self, inputs, *args, **kwargs):
    assert kwargs.get('get_all_layers', None), \
        "Only supporting returning all layers"
    single_frame_layers = self._get_single_frame_layers(inputs, *args, **kwargs)
    _, slow_layers = self.slow_model(inputs, *args, **kwargs)
    return None, [single_frame_layers, slow_layers]


class SlowFastSingleModel(SlowSingleModel):
  def __init__(self, multi_frame_choice='choose_one', *args, **kwargs):
    self.single_frame_model = SingleFrameModel(*args, **kwargs)
    self.slowfast_model = SlowFastModel(
        *args, **kwargs)
    self.multi_frame_choice = multi_frame_choice

  def __call__(self, inputs, slowfast_single_pooling=False, *args, **kwargs):
    assert kwargs.get('get_all_layers', None), \
        "Only supporting returning all layers"
    single_frame_layers = self._get_single_frame_layers(inputs, slowfast_single_pooling, *args, **kwargs)
    _, slowfast_layers = self.slowfast_model(inputs, slowfast_single_pooling, *args, **kwargs)
    return None, [single_frame_layers, slowfast_layers]
