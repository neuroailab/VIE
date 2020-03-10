from basics import basic
from vd_single_frame import vd_basic_color
from vd_slow import slow_bs


def vd_bin_slow_basic(args):
    args = vd_basic_color(args)
    args['data_len'] = 1145906
    args['bin_interval'] = 42
    return args


def vd_bin_slow_IR():
    args = {}

    args = basic(args)
    args = slow_bs(args)
    args = vd_bin_slow_basic(args)

    args['exp_id'] = 'vd_bin_slow_IR'
    args['model_type'] = 'slow'
    args['task'] = 'IR'
    return args
