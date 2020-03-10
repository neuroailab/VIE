from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf


class MemoryBank(object):
    def __init__(self, size, dim, seed=None):
        self.size = size
        self.dim = dim
        self.seed = seed or 0
        self._bank = self._create()

    def _create(self):
        mb_init = tf.random_uniform(
            shape=(self.size, self.dim),
            seed=self.seed,
        )
        std_dev = 1. / np.sqrt(self.dim/3)
        mb_init = mb_init * (2*std_dev) - std_dev
        return tf.get_variable(
            'memory_bank',
            initializer=mb_init,
            dtype=tf.float32,
            trainable=False,
        )

    def as_tensor(self):
        return self._bank

    def at_idxs(self, idxs):
        return tf.gather(self._bank, idxs, axis=0)

    def get_all_dot_products(self, vec):
        vec_shape = vec.get_shape().as_list()
        # [bs, dim]
        assert len(vec_shape) == 2
        return tf.matmul(vec, tf.transpose(self._bank, [1, 0]))

    def get_dot_products(self, vec, idxs):
        vec_shape = vec.get_shape().as_list()
        # [bs, dim]
        idxs_shape = idxs.get_shape().as_list()
        # [bs, ...]
        assert len(vec_shape) == 2
        assert vec_shape[0] == idxs_shape[0]

        memory_vecs = tf.gather(self._bank, idxs, axis=0)
        memory_vecs_shape = memory_vecs.get_shape().as_list()
        # [bs, ..., dim]
        assert memory_vecs_shape[:-1] == idxs_shape

        vec_shape[1:1] = [1] * (len(idxs_shape) - 1)
        vec = tf.reshape(vec, vec_shape)
        # [bs, 1,...,1, dim]

        prods = tf.multiply(memory_vecs, vec)
        assert prods.get_shape().as_list() == memory_vecs_shape
        return tf.reduce_sum(prods, axis=-1)
