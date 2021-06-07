import os
import sys
import pdb
import tensorflow as tf
from argparse import Namespace
from collections import OrderedDict
import model.instance_model as vd_inst_model
import numpy as np
SETTINGS = {
        '3dresnet': {
            'size': 112,
            'num_frames': 16},
        }


def get_network_outputs(
        input_images, 
        model_type):
    all_outs = vd_inst_model.resnet_embedding(
            input_images,
            get_all_layers='all_raw',
            skip_final_dense=True,
            model_type=model_type)
    return all_outs


def build_graph(model_type, batch_size):
    nf = SETTINGS[model_type]['num_frames']
    size = SETTINGS[model_type]['size']
    img_placeholder = tf.placeholder(
            dtype=tf.uint8, 
            shape=[batch_size, nf, size, size, 3])
    network_outputs = get_network_outputs(img_placeholder, model_type)
    return img_placeholder, network_outputs


def test_video_model():
    batch_size = 1
    img_placeholder, network_outputs = build_graph(
            model_type='3dresnet',
            batch_size=batch_size,
            )

    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(allow_growth=True)
    SESS = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=gpu_options,
            ))
    # Change this to your ckpt path
    model_ckpt_path = '/mnt/fs4/chengxuz/brainscore_model_caches/vd_unsup_fx/dyn_clstr/vd_3dresnet/checkpoint-1450000'
    # This should be the actual input clips
    input_images = np.zeros([batch_size, 16, 112, 112, 3], dtype=np.uint8)
    saver.restore(SESS, model_ckpt_path)
    outputs_np = SESS.run(
            network_outputs, 
            feed_dict={img_placeholder: input_images})
    pdb.set_trace()
    pass


if __name__ == '__main__':
    test_video_model()
