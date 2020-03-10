from basics import basic_fix, bs128
from vd_single_frame_fx import vd_basic


def vd_bin_basic(args):
    args = vd_basic(args)
    args['data_len'] = 1157650
    args['bin_interval'] = 52
    return args


def vd_ctl_bin_IR():
    args = {}

    args = basic_fix(args)
    args = bs128(args)
    args = vd_bin_basic(args)

    args['exp_id'] = 'vd_ctl_bin_IR'
    args['task'] = 'IR'
    args['lr_boundaries'] = '1125011,1535011'
    return args


def load_from_bin_IR(args):
    args['load_exp'] = 'vd_unsup_fx/dyn_clstr/vd_ctl_bin_IR'
    args['load_step'] = 50000
    return args


def vd_ctl_bin():
    args = {}

    args = basic_fix(args)
    args = bs128(args)
    args = vd_bin_basic(args)
    args = load_from_bin_IR(args)

    args['exp_id'] = 'vd_ctl_bin'
    args['kmeans_k'] = '10000'
    args['lr_boundaries'] = '1220011,1505011'
    return args


def vd_big_bin_basic(args):
    args = vd_basic(args)
    args['data_len'] = 473532
    args['bin_interval'] = 130
    return args


def vd_ctl_big_bin_IR():
    args = {}

    args = basic_fix(args)
    args = bs128(args)
    args = vd_big_bin_basic(args)

    args['exp_id'] = 'vd_ctl_big_bin_IR'
    args['task'] = 'IR'
    args['lr_boundaries'] = '1350011,1960011'
    return args


def load_from_big_bin_IR(args):
    args['load_exp'] = 'vd_unsup_fx/dyn_clstr/vd_ctl_big_bin_IR'
    args['load_step'] = 50000
    return args


def vd_ctl_big_bin():
    args = {}

    args = basic_fix(args)
    args = bs128(args)
    args = vd_big_bin_basic(args)
    args = load_from_big_bin_IR(args)

    args['exp_id'] = 'vd_ctl_big_bin'
    args['kmeans_k'] = '5000'
    args['lr_boundaries'] = '1375011,1600011'
    return args
