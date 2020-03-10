from __future__ import division, print_function
import os, sys
import numpy as np
import tensorflow as tf

from .resnet_th_preprocessing import (
    preprocessing_inst, RandomSizedCrop_from_jpeg,
    ApplyGray, ColorJitter, alexnet_crop_from_jpg
)

# This file contains various preprocessing ops for images (typically
# used for data augmentation).

def resnet_train(img_str):
    return preprocessing_inst(img_str, 224, 224, is_train=True)


def resnet_train_112(img_str):
    return preprocessing_inst(img_str, 112, 112, is_train=True)


def resnet_validate(img_str):
    return preprocessing_inst(img_str, 224, 224, is_train=False)


def resnet_validate_112(img_str):
    return preprocessing_inst(
            img_str, 112, 112, 
            is_train=False, val_short_side=128)


def resnet_crop_only(img_str):
    return RandomSizedCrop_from_jpeg(
        img_str, out_height=224, out_width=224, size_minval=0.2)


def resnet_crop_flip(img_str):
    img = RandomSizedCrop_from_jpeg(
            img_str, out_height=224, out_width=224, size_minval=0.2)
    img = tf.image.random_flip_left_right(img)
    return img


def alexnet_crop_flip(img_str):
    img = alexnet_crop_from_jpg(img_str)
    img = tf.image.random_flip_left_right(img)
    return img


def resnet_noflip(img_str):
    img = resnet_crop_only(img_str)
    img = ApplyGray(img, 0.2)
    return ColorJitter(img)


def resnet_nocrop(img_str):
    img = resnet_validate(img_str)
    img = ApplyGray(img, 0.2)
    img = ColorJitter(img)
    return tf.image.random_flip_left_right(img)


def resnet_bigcrop(img_str):
    return preprocessing_inst(img_str, 224, 224, is_train=True,
                              size_minval=0.6)


def resnet_4way_rot(img_str):
    img = resnet_train(img_str)
    angle_choice = tf.random_uniform([], maxval=4, dtype=tf.int32)
    angle = tf.cast(angle_choice, tf.float32) * (np.pi/2)
    img = tf.contrib.image.rotate(img, angle)
    return img


def resnet_rot(img_str):
    img = resnet_train(img_str)
    angle_choice = tf.random_uniform([], maxval=1, dtype=tf.float32)
    angle = angle_choice * (2*np.pi)
    img = tf.contrib.image.rotate(img, angle)
    return img


def _get_resize_scale(height, width, smallest_side):
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)

    scale = tf.cond(
            tf.greater(height, width),
            lambda: smallest_side / width,
            lambda: smallest_side / height)
    return scale


def center_crop(img_str, out_height, out_width):
    shape = tf.image.extract_jpeg_shape(image_string)
    # the scaling factor needed to make the smaller side 256
    scale = _get_resize_scale(shape[0], shape[1], 256)
    cp_height = tf.cast(out_height / scale, tf.int32)
    cp_width = tf.cast(out_width / scale, tf.int32)
    cp_begin_x = tf.cast((shape[0] - cp_height) / 2, tf.int32)
    cp_begin_y = tf.cast((shape[1] - cp_width) / 2, tf.int32)
    bbox = tf.stack([cp_begin_x, cp_begin_y,
                     cp_height, cp_width])
    crop_image = tf.image.decode_and_crop_jpeg(
        image_string, bbox, channels=3)
    image = image_resize(crop_image, out_height, out_width)

    image.set_shape([out_height, out_width, 3])
    return image


def rgb_to_gray(flt_image):
    flt_image = tf.cast(flt_image, tf.float32)
    gry_image = flt_image[:,:,0] * 0.299 \
            + flt_image[:,:,1] * 0.587 \
            + flt_image[:,:,2] * 0.114
    gry_image = tf.expand_dims(gry_image, axis=2)
    gry_image = tf.cast(gry_image + EPS, tf.uint8)
    gry_image = tf.cast(gry_image, tf.float32)
    return gry_image
