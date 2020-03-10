from __future__ import division, print_function, absolute_import
import os, sys
import functools
import numpy as np
import tensorflow as tf


def image_dir_to_tfrecords_dataset(image_dir, is_train):
    pattern = 'train-*' if is_train else 'validation-*'
    pattern = os.path.join(image_dir, pattern)
    datasource = tf.gfile.Glob(pattern)
    datasource.sort()
    tfr_list = np.asarray(datasource)
    dataset = tf.data.Dataset.list_files(tfr_list)

    if is_train:
        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(len(tfr_list))
        )
    else:
        dataset = dataset.repeat()

    def fetch(filename):
        buffer_size = 32 * 1024 * 1024 # 32 MiB per file
        return tf.data.TFRecordDataset(filename, buffer_size=buffer_size)

    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            fetch, cycle_length=8, sloppy=True))
    return dataset


def data_parser(record_str_tensor, process_img_func,
                is_train=True, with_indx=True, num_tile=None):
    '''
    Takes a TFRecord string and outputs a dictionary ready to use
    as input to the model.
    '''

    # Parse the TFRecord
    keys_to_features = {
            'images': tf.FixedLenFeature((), tf.string, ''),
            'labels': tf.FixedLenFeature([], tf.int64, -1)}
    if with_indx:
        keys_to_features['index'] = tf.FixedLenFeature([], tf.int64, -1)
    parsed = tf.parse_single_example(record_str_tensor, keys_to_features)
    image_string = parsed['images']
    image_label = parsed['labels']
    image_index = parsed.get('index', None)

    # Process the image
    image = process_img_func(image_string)
    if num_tile is not None:
        curr_shape = image.get_shape().as_list()
        image = tf.expand_dims(image, axis=0)
        image = tf.tile(image, [num_tile] + [1] * len(curr_shape))
    ret_dict = {'image': image, 'label': image_label}
    if with_indx:
        ret_dict['index'] = image_index
    return ret_dict


def dataset_func(
        image_dir, process_img_func, is_train, batch_size, q_cap, 
        num_tile=None):
    dataset = image_dir_to_tfrecords_dataset(image_dir, is_train=is_train)
    if is_train:
        dataset = dataset.shuffle(buffer_size=q_cap)
    dataset = dataset.prefetch(batch_size * 4)
    dataset = dataset.map(functools.partial(
            data_parser, process_img_func=process_img_func,
            is_train=is_train, with_indx=is_train,
            num_tile=num_tile,
        ), num_parallel_calls=64)
    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.prefetch(4)
    next_element = dataset.make_one_shot_iterator().get_next()
    return next_element
