import time
import faiss
import numpy as np
import tensorflow as tf

DEFAULT_SEED = 1234


def run_kmeans(x, nmb_clusters, verbose=False, seed=DEFAULT_SEED):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    clus.seed = seed
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    losses = faiss.vector_to_array(clus.obj)
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses[-1]


class Kmeans:
    def __init__(self, k, memory_bank, cluster_labels):
        self.k = k
        self.memory_bank = memory_bank
        self.cluster_labels = cluster_labels

        self.new_cluster_feed = tf.placeholder(
            tf.int64, shape=self.cluster_labels.get_shape().as_list())
        self.update_clusters_op = tf.assign(
                self.cluster_labels, self.new_cluster_feed)

    def recompute_clusters(self, sess, verbose=True):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        data = sess.run(self.memory_bank.as_tensor())

        all_lables = []
        for k_idx, each_k in enumerate(self.k):
            # cluster the data
            I, _ = run_kmeans(data, each_k, 
                              verbose, seed = k_idx + DEFAULT_SEED)
            new_clust_labels = np.asarray(I)
            all_lables.append(new_clust_labels)
        new_clust_labels = np.stack(all_lables, axis=0)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))
        return new_clust_labels

    def apply_clusters(self, sess, new_clust_labels):
        sess.run(self.update_clusters_op, feed_dict={
            self.new_cluster_feed: new_clust_labels
        })
