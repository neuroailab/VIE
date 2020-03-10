from saved_settings.basics import basic_fix
from saved_settings.vd_single_frame_fx import vd_basic
from saved_settings.vd_slow_fx import slow_bs


def vd_3dresnet_IR():
    args = {}

    args = basic_fix(args)
    args = slow_bs(args)
    args = vd_basic(args)
    args['train_prep'] = 'ColorJitter_112'
    args['port'] = 27007

    args['exp_id'] = 'vd_3dresnet_IR'
    args['model_type'] = '3dresnet'
    args['task'] = 'IR'
    args['train_num_workers'] = 40
    args['val_num_workers'] = 20
    args['train_num_steps'] = 50000
    return args


def load_from_3dresnet_IR(args):
    args['load_exp'] = 'vd_unsup_fx/dyn_clstr/vd_3dresnet_IR'
    args['load_step'] = 50000
    return args


def vd_3dresnet():
    args = {}

    args = basic_fix(args)
    args = slow_bs(args)
    args = vd_basic(args)
    args = load_from_3dresnet_IR(args)
    args['train_prep'] = 'ColorJitter_112'
    args['port'] = 27007

    args['exp_id'] = 'vd_3dresnet'
    args['model_type'] = '3dresnet'
    args['lr_boundaries'] = '974946,1304998'
    args['train_num_steps'] = 1600000
    args['train_num_workers'] = 40
    args['val_num_workers'] = 20
    return args
