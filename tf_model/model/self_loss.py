from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf

DATA_LEN_IMAGENET_FULL = 1281167


def assert_shape(t, shape):
    assert t.get_shape().as_list() == shape, \
            "Got shape %r, expected %r" % (t.get_shape().as_list(), shape)


def get_selfloss(memory_bank, **kwargs):
    use_clusters = kwargs.get('use_clusters')
    if use_clusters is not None:
        all_labels = np.load(use_clusters) # path to all labels
        assert len(all_labels.shape) == 1
        sample_num = kwargs.get('num_cluster_samples')
        print("Using %d cluster labels read from %s for self-loss" %
              (len(all_labels), use_clusters))
        print("Sampling %d items from the cluster each time" % sample_num)
        return ClusterLoss(memory_bank, all_labels, sample_num=sample_num)

    add_topn_dot = kwargs.get('add_topn_dot')
    add_thres_dot = kwargs.get('add_thres_dot')
    nn_list_path = kwargs.get('nn_list_path')
    if add_topn_dot is not None and add_thres_dot is not None:
        raise NotImplementedError(
            "Add_topn_dot and add_thres_dot cannot be both on!")

    if add_topn_dot is not None:
        if nn_list_path is not None:
            print("Using nearest %i self-loss with fixed "
                  "neighbors loaded from %s" % (add_topn_dot, nn_list_path))
            nn = np.load(nn_list_path)
            nn = nn['highest_dp_indices'][:,:add_topn_dot]
            print("Loaded nearest neighbor indices with shape", nn.shape)
            return StaticNearestNeighborLoss(memory_bank, nn)

        print("Using nearest %i self-loss." % add_topn_dot)
        return NearestNLoss(memory_bank, add_topn_dot)

    elif add_thres_dot is not None:
        print("Using threshold self-loss with threshold %f." % add_thres_dot)
        return ThresholdNeighborLoss(memory_bank, add_thres_dot)

    return DefaultSelfLoss(memory_bank)


class DefaultSelfLoss(object):
    def __init__(self, memory_bank):
        self.memory_bank = memory_bank

    def get_closeness(self, idxs, vecs):
        return self.memory_bank.get_dot_products(vecs, idxs)


class StaticNearestNeighborLoss(object):
    def __init__(self, memory_bank, nearest_neighbors):
        self.memory_bank = memory_bank
        nn_shape = nearest_neighbors.shape
        # [data_len, num_neighbors]
        # Your nearest neighbor is yourself, so if num_neighbors is 1, this
        # should be equivalent to the default loss.
        assert len(nn_shape) == 2
        self.nn = tf.constant(nearest_neighbors)

    def get_closeness(self, idxs, vecs):
        '''
        idxs: The indices whose neighbors we care about.
        vecs: The embedding values of those indices.
        '''
        cur_nn = tf.gather(self.nn, idxs, axis=0)
        nn_dps = self.memory_bank.get_dot_products(vecs, cur_nn)
        return tf.reduce_mean(nn_dps, axis=-1)


class NearestNLoss(object):
    def __init__(self, memory_bank, n):
        self.memory_bank = memory_bank
        self.n = n

    def get_closeness(self, idxs, vecs):
        batch_size = idxs.get_shape().as_list()[0]
        all_dps = self.memory_bank.get_all_dot_products(vecs)
        topn_values, _ = tf.nn.top_k(all_dps, k=self.n, sorted=False)
        assert_shape(topn_values, [batch_size, self.n])
        return tf.reduce_mean(topn_values, axis=1)


class ThresholdNeighborLoss(object):
    def __init__(self, memory_bank, treshold):
        self.memory_bank = memory_bank
        self.treshold = treshold

    def get_closeness(self, idxs, vecs):
        batch_size = idxs.get_shape().as_list()[0]

        # Currently take first 1000 and then threshold it
        # TODO: fix this to be more general
        all_dps = self.memory_bank.get_all_dot_products(vecs)
        topn_values, _ = tf.nn.top_k(all_dps, k=1000, sorted=False)

        big_mask = topn_values > self.treshold
        all_values_under_mask = tf.boolean_mask(topn_values, big_mask)
        all_indexes_for_mask = tf.where(big_mask)
        ## As we only need the batch dimension
        batch_indexes_for_mask = all_indexes_for_mask[:, 0]
        big_number = tf.unsorted_segment_sum(
            tf.ones_like(all_values_under_mask),
            batch_indexes_for_mask,
            batch_size,
        )
        big_sum = tf.unsorted_segment_sum(
            all_values_under_mask,
            batch_indexes_for_mask,
            batch_size,
        )

        # Add the original data dot-product in case the threshold too high
        data_dot_product = self.memory_bank.get_dot_products(vecs, idxs)
        big_sum += data_dot_product
        big_number += 1
        return tf.reshape(big_sum/big_number, [batch_size])


class ClusterLoss(object):
    @staticmethod
    def pad_clusters_to_same_size(clusters):
        '''
        Make clusters the same size by repeating elements.
        '''
        ret = []
        max_size = max(len(c) for c in clusters)
        for c in clusters:
            c = np.array(c)
            tiling = np.tile(c,  (max_size // len(c)))
            # TODO: consider setting numpy seed
            padding = np.random.choice(
                c, size=(max_size - len(tiling)), replace=False)
            ret.append(np.concatenate([tiling, padding]))
        return np.stack(ret)

    def __init__(self, memory_bank, cluster_labels, sample_num=None):
        self.memory_bank = memory_bank
        self.cluster_labels = cluster_labels
        self.n = np.max(cluster_labels) + 1
        print('Initializing cluster loss with %i clusters' % self.n)
        # number of same-cluster labels to sample, or None meaning take
        # all of them
        self.sample_num = sample_num

        self.clusters = [[] for _ in range(self.n)]
        for idx, label in enumerate(self.cluster_labels):
            self.clusters[label].append(idx)
        self.clusters = ClusterLoss.pad_clusters_to_same_size(self.clusters)
        _, self.max_cluster_size = self.clusters.shape
        print('Padding each cluster to size %i' % self.max_cluster_size)

    def get_closeness(self, idxs, vecs):
        batch_size = idxs.get_shape().as_list()[0]
        cluster_ids = tf.gather(self.cluster_labels, idxs)
        # [bs]

        if self.sample_num is None:
            # Don't sample
            cluster_lists = tf.gather(self.clusters, cluster_ids)
            assert_shape(cluster_lists, [batch_size, self.max_cluster_size])
            cluster_dps = self.memory_bank.get_dot_products(vecs, cluster_lists)
            return tf.reduce_mean(cluster_dps, axis=1)

        same_clust_idxs = tf.random_uniform(
            shape=(batch_size, self.sample_num),
            minval=0, maxval=self.max_cluster_size,
            dtype=tf.int64
        ) # now indices into the cluster lists
        same_clust_idxs = tf.stack([
            tf.stack([cluster_ids] * self.sample_num, axis=-1),
            same_clust_idxs
        ], axis=-1)
        assert_shape(same_clust_idxs, [batch_size, self.sample_num, 2])
        # [bs, sample_num, 2], stack with labels in preparation for
        # gather_nd, each index term [a, b] refers to
        # self.clusters[a][b]
        same_clust_idxs = tf.gather_nd(self.clusters, same_clust_idxs)
        assert_shape(same_clust_idxs, [batch_size, self.sample_num])
        # [bs, sample_num]

        dps = self.memory_bank.get_dot_products(
            vecs, same_clust_idxs)
        return tf.reduce_mean(dps, axis=1)
