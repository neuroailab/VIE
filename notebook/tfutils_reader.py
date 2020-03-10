from __future__ import division, print_function, absolute_import

import pymongo as pm
import gridfs
from tensorflow.core.protobuf import saver_pb2
import tarfile
import cPickle

import numpy as np
from scipy import misc
import os
import time
import sklearn.linear_model
import math

import tensorflow as tf
from tfutils.db_interface import verify_pb2_v2_files


def _print_checkpt_vars(path):
    # For debugging
    from tensorflow.python.tools.inspect_checkpoint import (
        print_tensors_in_checkpoint_file
    )
    print_tensors_in_checkpoint_file(path,
                                     all_tensor_names=True,
                                     all_tensors=False,
                                     tensor_name='')


class TfutilsReader(object):
    def __init__(self, dbname, colname, exp_id,
                 port, cache_dir):
        self.exp_id = exp_id
        self.conn = conn = pm.MongoClient(port=port)

        self.coll = conn[dbname][colname + '.files']
        self.collfs = gridfs.GridFS(conn[dbname], colname)
        self.fs_bucket = gridfs.GridFSBucket(conn[dbname], colname)

        self.load_files_dir = os.path.join(cache_dir, dbname, colname, exp_id)

    def query(self, query_dict, restrict_fields=None, **kwargs):
        # commonly used kwargs: sort, projection
        query_dict = query_dict.copy()
        query_dict['exp_id'] = self.exp_id
        if restrict_fields is None:
            return self.coll.find(query_dict, **kwargs)
        return self.coll.find(query_dict, restrict_fields, **kwargs)

    def load_gridfs_file(self, rec):
        '''
        Converts a GridFS file to an ordinary file and returns the
        path where the GridFS contents were copied.
        '''
        assert 'saved_filters' in rec

        if not os.path.exists(self.load_files_dir):
            os.makedirs(self.load_files_dir)
        fname = os.path.basename(rec['filename'])
        path = os.path.join(self.load_files_dir, fname)

        if rec['_saver_write_version'] == saver_pb2.SaverDef.V2:
            extracted_path = os.path.splitext(path)[0]
            if os.path.exists(extracted_path + '.index'):
                print('Using already present file at extraction path %s.'
                      % extracted_path)
                return extracted_path
        elif os.path.exists(path):
            print('Using already present file at extraction path %s.' % path)
            return path

        fs_file = open(path, 'wrb+')
        self.fs_bucket.download_to_stream(rec['_id'], fs_file)
        fs_file.close()

        if rec['_saver_write_version'] == saver_pb2.SaverDef.V2:
            assert fname.endswith('.tar')
            tar = tarfile.open(path)
            tar.extractall(path=self.load_files_dir)
            tar.close()
            path = os.path.splitext(path)[0]
            verify_pb2_v2_files(path, rec)
        return path
