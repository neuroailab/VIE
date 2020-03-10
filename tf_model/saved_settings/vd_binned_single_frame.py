from basics import basic, bs128
from vd_single_frame import vd_basic_color


def vd_bin_basic(args):
    args = vd_basic_color(args)
    args['data_len'] = 1157650
    args['bin_interval'] = 52
    return args


def vd_ctl_bin_IR():
    args = {}

    args = basic(args)
    args = bs128(args)
    args = vd_bin_basic(args)

    args['exp_id'] = 'vd_ctl_bin_IR'
    args['task'] = 'IR'
    return args


def load_from_bin_IR(args):
    args['load_exp'] = 'vd_unsup/dyn_clstr/vd_ctl_bin_IR'
    args['load_step'] = 50000
    return args


def vd_ctl_bin():
    args = {}

    args = basic(args)
    args = bs128(args)
    args = vd_bin_basic(args)
    args = load_from_bin_IR(args)

    args['exp_id'] = 'vd_ctl_bin'
    args['kmeans_k'] = '10000'
    return args
