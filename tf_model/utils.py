from __future__ import division, print_function, absolute_import
import os, sys, datetime
import numpy as np
import tensorflow as tf
import copy
import pdb
from model.self_loss import DATA_LEN_IMAGENET_FULL, assert_shape
DATA_LEN_KINETICS_400 = 239888
VAL_DATA_LEN_KINETICS_400 = 19653


def online_keep_all(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(v)
    return agg_res


def tuple_get_one(x):
    if isinstance(x, tuple) or isinstance(x, list):
        return x[0]
    return x
