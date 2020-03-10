import sys
import os
import tensorflow as tf
import pdb

sys.path.append(os.path.abspath('../'))
from tf_model.model import resnet_model_slowfast as sf_model


SINGLE_MODEL_PATH = '/data/chengxuz/.tfutils/localhost:27006/vd_unsup_fx/dyn_clstr/vd_ctl/checkpoint-1375000'
SLOW_MODEL_PATH = '/data/chengxuz/.tfutils/localhost:27006/vd_unsup_fx/dyn_clstr/vd_slow/checkpoint-1440000'
SLOWFAST_MODEL_PATH = '/data/chengxuz/.tfutils/localhost:27006/vd_unsup_fx/dyn_clstr/vd_slowfast_a4/checkpoint-1350000'
OUTPUT_DIR = '/mnt/fs3/chengxuz/vd_relat/slow_single_model'


def main():
    os.system('mkdir -p %s' % OUTPUT_DIR)

    slow_single_model = sf_model.SlowSingleModel(resnet_size=18)
    input_image = tf.zeros([1, 4, 224, 224, 3], dtype=tf.float32)
    _, _ = slow_single_model(input_image, False, get_all_layers=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    single_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_model/')
    single_saver = tf.train.Saver(single_vars)

    slow_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_model_slow/')
    var_dict = {}
    for each_var in slow_vars:
        new_name = each_var.op.name.replace(
                'resnet_model_slow/', 'resnet_model/')
        var_dict[new_name] = each_var
    slow_saver = tf.train.Saver(var_dict)

    final_saver = tf.train.Saver()

    #reader = tf.train.NewCheckpointReader(SLOW_MODEL_PATH)
    #var_shapes = reader.get_variable_to_shape_map()
    #print('Saved vars and shapes:\n' + str(var_shapes))

    with tf.Session() as sess:
        single_saver.restore(sess, SINGLE_MODEL_PATH)
        slow_saver.restore(sess, SLOW_MODEL_PATH)
        assert len(sess.run(tf.report_uninitialized_variables())) == 0
        final_saver.save(sess, os.path.join(OUTPUT_DIR, 'model'))


OUTPUT_SF_DIR = '/mnt/fs3/chengxuz/vd_relat/slowfast_single_model'


def main_sf():
    os.system('mkdir -p %s' % OUTPUT_SF_DIR)

    slowfast_single_model = sf_model.SlowFastSingleModel(resnet_size=18)
    input_image = tf.zeros([1, 16, 224, 224, 3], dtype=tf.float32)
    _, _ = slowfast_single_model(input_image, False, get_all_layers=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    single_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_model/')
    single_saver = tf.train.Saver(single_vars)

    slowfast_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_model_')
    slowfast_saver = tf.train.Saver(slowfast_vars)

    final_saver = tf.train.Saver()

    with tf.Session() as sess:
        single_saver.restore(sess, SINGLE_MODEL_PATH)
        slowfast_saver.restore(sess, SLOWFAST_MODEL_PATH)
        assert len(sess.run(tf.report_uninitialized_variables())) == 0
        final_saver.save(sess, os.path.join(OUTPUT_SF_DIR, 'model'))


if __name__ == "__main__":
    #main()
    main_sf()
