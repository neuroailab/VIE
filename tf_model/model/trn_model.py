from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pdb
import itertools


class Relation(object):

    def __init__(
            self, num_inputs, 
            out_features, 
            bottleneck_dim=512,
            layer_name=None):
        self.num_inputs = num_inputs
        self.out_features = out_features
        self.bottleneck_dim = bottleneck_dim
        if layer_name is not None:
            self.layer_name = layer_name
        else:
            self.layer_name = 'relation_{}'.format(num_inputs)

    def __call__(self, inputs):
        no_rel, bs, _, _ = inputs.get_shape().as_list()
        inputs = tf.reshape(inputs, [no_rel * bs, -1])
        inputs = tf.layers.dense(inputs=inputs, units=self.bottleneck_dim,
                                 activation=tf.nn.relu,
                                 name=self.layer_name + '_hidden')
        inputs = tf.layers.dense(inputs=inputs, units=self.out_features,
                                 name=self.layer_name + '_out')
        inputs = tf.reshape(inputs, [no_rel, bs, -1])
        return inputs


class MultiScaleRelation(object):
    """Multi-Relation module.
    This module applies mlps to concatenated n-input tuples.

    Args:
        num_frame_total: total number of frame features (e.g. 16 frames).
        out_features: dim of output relation feature.
        bottleneck_dim: dim of bottleneck in each relation MLP.
        num_relations: number of MLPs used for frame relation tuples.

    """

    def __init__(self,
                 num_frame_total,
                 out_features,
                 bottleneck_dim=512,
                 num_relations=8,
                 use_mean=False):
        self.num_frame_total = num_frame_total
        self.out_features = out_features
        self.num_relations = num_relations
        self.bottleneck_dim = bottleneck_dim
        self.use_mean = use_mean

        self.scales = list(range(num_frame_total, 1, -1))
        self.relations_scales = []
        self.subsample_scales = []

        for scale in self.scales:

            # Determine possible `scale`-input tuples e.g:
            # [(0, 1), (0, 2), (1, 2)] =  self.return_relationset(2)
            relations_scale = self.return_relationset(scale)
            self.relations_scales.append(relations_scale)

            # Limit number of
            self.subsample_scales.append(
                min(self.num_relations, len(relations_scale)))

        # Each Relation takes `scale` num frame features.
        self.relations = [  
                Relation(scale, self.out_features, self.bottleneck_dim) \
                for scale in self.scales]

        print('Adding multi-Scale Relation Network Module')
        print(['{}-frame relation'.format(i) for i in self.scales])

    def return_relationset(self, num_input_relation):
        return list(itertools.combinations(
            range(self.num_frame_total), num_input_relation))

    def __call__(self, input):
        """Apply TRN module.

        Args:
            input: frame features (batch_size, num_frames, feature_dim)

        Returns:
            video embedding: (batch_size, out_features)
        """
        output = []
        for idx_scale, scale in enumerate(self.scales):
            curr_num_relations = self.subsample_scales[idx_scale]
            max_num_relations = len(self.relations_scales[idx_scale])
            idx_relations = tf.random_uniform(
                    shape=[curr_num_relations],
                    minval=0, maxval=max_num_relations,
                    dtype=tf.int64)

            input_to_mlp = []
            curr_relation_ts = tf.constant(self.relations_scales[idx_scale])
            for idx in range(curr_num_relations):
                curr_idx_rel = idx_relations[idx]
                curr_rel_tuple = curr_relation_ts[curr_idx_rel]
                input_relation = tf.gather(input, curr_rel_tuple, axis=1)
                input_to_mlp.append(input_relation)
            input_to_mlp = tf.stack(input_to_mlp, axis=0)
            output_of_mlp = self.relations[idx_scale](input_to_mlp)
            output.append(output_of_mlp)

        output = tf.concat(output, axis=0)
        if not self.use_mean:
            output = tf.reduce_sum(output, axis=0)
        else:
            output = tf.reduce_mean(output, axis=0)
        return output
