from basics import basic
from vd_single_frame import vd_basic_color, res18_la_one_bigk, \
        load_from_IR_color, res18_la_one_bigk_smN, res18_la_one_bigk_ssmN


def fast_bs(args):
    args['batch_size'] = 64
    args['test_batch_size'] = 32
    args['test_no_frames'] = 5
    args['kNN_val'] = 10
    args['fre_filter'] = 50000
    args['fre_cache_filter'] = 5000
    args['fre_valid'] = 5000
    return args


def vd_fast_color_IR():
    args = {}

    args = basic(args)
    args = fast_bs(args)
    args = vd_basic_color(args)

    args['exp_id'] = 'vd_fast_color_IR_fx'
    args['model_type'] = 'fast'
    args['task'] = 'IR'
    args['train_num_workers'] = 30
    return args


def fast_a4(args):
    args['model_type'] = 'fast_a4'
    args['train_num_workers'] = 30
    args['val_num_workers'] = 10
    return args


def vd_fast_a4_color_IR():
    args = {}

    args = basic(args)
    args = fast_bs(args)
    args = vd_basic_color(args)
    args = fast_a4(args)

    args['exp_id'] = 'vd_fast_a4_color_IR'
    args['task'] = 'IR'
    args['lr_boundaries'] = '325011'
    return args


def load_from_fast_IR(args):
    args['load_exp'] = 'vd_unsup/dyn_clstr/vd_fast_a4_color_IR'
    args['load_step'] = 50000
    return args


def vd_fast_a4_LA():
    args = {}

    args = basic(args)
    args = fast_bs(args)
    args = vd_basic_color(args)
    args = load_from_fast_IR(args)
    args = res18_la_one_bigk_smN(args)
    args = fast_a4(args)

    args['exp_id'] = 'vd_fast_a4_LA'
    return args


def vd_fast_a4_LA_ssmN():
    args = {}

    args = basic(args)
    args = fast_bs(args)
    args = vd_basic_color(args)
    args = load_from_fast_IR(args)
    args = res18_la_one_bigk_ssmN(args)
    args = fast_a4(args)

    args['exp_id'] = 'vd_fast_a4_LA_ssmN'
    return args
