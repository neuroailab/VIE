from basics import basic_fix
from vd_single_frame_fx import vd_basic, res18_la
from vd_slow_fx import slow_bs


def vd_trn_IR():
    args = {}

    args = basic_fix(args)
    args = slow_bs(args)
    args = vd_basic(args)

    args['exp_id'] = 'vd_trn_IR'
    args['model_type'] = 'trn'
    args['task'] = 'IR'
    return args


def vd_trn_f4_IR():
    args = {}

    args = basic_fix(args)
    args = slow_bs(args)
    args = vd_basic(args)

    args['exp_id'] = 'vd_trn_f4_IR'
    args['model_type'] = 'trn'
    args['task'] = 'IR'
    args['trn_num_frames'] = 4
    return args


def load_from_LA_final(args):
    args['load_exp'] = 'vd_unsup_fx/dyn_clstr/vd_ctl'
    return args


def vd_trn_f4_pret():
    args = {}

    args = basic_fix(args)
    args = slow_bs(args)
    args = vd_basic(args)
    args = load_from_LA_final(args)
    args = res18_la(args)

    args['exp_id'] = 'vd_trn_f4_pret'
    args['model_type'] = 'trn'
    args['trn_num_frames'] = 4
    args['lr_boundaries'] = '1460011,1570011'
    return args


def vd_trn_pret():
    args = {}

    args = basic_fix(args)
    args = slow_bs(args)
    args = vd_basic(args)
    args = load_from_LA_final(args)
    args = res18_la(args)

    args['exp_id'] = 'vd_trn_pret'
    args['model_type'] = 'trn'
    args['lr_boundaries'] = '1460011,1570011'
    return args


def vd_tsrn_IR():
    args = {}

    args = basic_fix(args)
    args = slow_bs(args)
    args = vd_basic(args)

    args['exp_id'] = 'vd_tsrn_IR'
    args['model_type'] = 'tsrn'
    args['task'] = 'IR'
    return args


def vd_tsrn_f4_IR():
    args = {}

    args = basic_fix(args)
    args = slow_bs(args)
    args = vd_basic(args)

    args['exp_id'] = 'vd_tsrn_f4_IR'
    args['model_type'] = 'tsrn'
    args['task'] = 'IR'
    args['trn_num_frames'] = 4
    return args


def load_from_tsrn_f4_IR(args):
    args['load_exp'] = 'vd_unsup_fx/dyn_clstr/vd_tsrn_f4_IR'
    args['load_step'] = 50000
    return args


def vd_tsrn_f4():
    args = {}

    args = basic_fix(args)
    args = slow_bs(args)
    args = vd_basic(args)
    args = load_from_tsrn_f4_IR(args)
    args = res18_la(args)

    args['exp_id'] = 'vd_tsrn_f4'
    args['model_type'] = 'tsrn'
    args['trn_num_frames'] = 4
    return args


def vd_tsrn_f4_pret():
    args = {}

    args = basic_fix(args)
    args = slow_bs(args)
    args = vd_basic(args)
    args = load_from_LA_final(args)
    args = res18_la(args)

    args['exp_id'] = 'vd_tsrn_f4_pret'
    args['model_type'] = 'tsrn'
    args['trn_num_frames'] = 4
    args['lr_boundaries'] = '1455011,1570011'
    return args


def vd_tsrn_slow_IR():
    args = {}

    args = basic_fix(args)
    args = slow_bs(args)
    args = vd_basic(args)

    args['exp_id'] = 'vd_tsrn_slow_IR'
    args['model_type'] = 'tsrn_slow'
    args['task'] = 'IR'
    return args


def load_from_tsrn_slow_IR(args):
    args['load_exp'] = 'vd_unsup_fx/dyn_clstr/vd_tsrn_slow_IR'
    args['load_step'] = 50000
    return args


def vd_tsrn_slow():
    args = {}

    args = basic_fix(args)
    args = slow_bs(args)
    args = vd_basic(args)
    args = load_from_tsrn_slow_IR(args)
    args = res18_la(args)

    args['exp_id'] = 'vd_tsrn_slow'
    args['model_type'] = 'tsrn_slow'
    return args


def vd_tsrn_slow_pret():
    args = {}

    args = basic_fix(args)
    args = slow_bs(args)
    args = vd_basic(args)
    args = load_from_LA_final(args)
    args = res18_la(args)

    args['exp_id'] = 'vd_tsrn_slow_pret'
    args['model_type'] = 'tsrn_slow'
    args['lr_boundaries'] = '1455011,1570011'
    return args
