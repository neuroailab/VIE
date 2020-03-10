from basics import basic_fix
from vd_single_frame_fx import vd_basic, res18_la
from vd_slow_fx import slow_bs


def slowfast_a4_basic(args):
    args['model_type'] = 'slowfast_a4'
    args['train_num_workers'] = 30
    args['val_num_workers'] = 10
    return args


def vd_slowfast_a4_IR():
    args = {}

    args = basic_fix(args)
    args = slow_bs(args)
    args = vd_basic(args)
    args = slowfast_a4_basic(args)

    args['exp_id'] = 'vd_slowfast_a4_IR'
    args['task'] = 'IR'
    return args


def load_from_slowfast_a4_IR(args):
    args['load_exp'] = 'vd_unsup_fx/dyn_clstr/vd_slowfast_a4_IR'
    args['load_step'] = 50000
    return args


def vd_slowfast_a4():
    args = {}

    args = basic_fix(args)
    args = slow_bs(args)
    args = vd_basic(args)
    args = load_from_slowfast_a4_IR(args)
    args = res18_la(args)
    args = slowfast_a4_basic(args)

    args['exp_id'] = 'vd_slowfast_a4'
    args['lr_boundaries'] = '1120011,1294998'
    return args


def vd_slowfast_a4_test():
    args = {}

    args = basic_fix(args)
    args = slow_bs(args)
    args = vd_basic(args)
    args = res18_la(args)
    args = slowfast_a4_basic(args)

    args['load_exp'] = 'vd_unsup_fx/dyn_clstr/vd_slowfast_a4'
    args['exp_id'] = 'vd_slowfast_a4_test'
    args['lr_boundaries'] = '1120011,1294998'
    args['plot_val'] = True
    args['pure_test'] = True
    return args


def vd_slowfast_a4_test_noimg():
    args = {}

    args = basic_fix(args)
    args = slow_bs(args)
    args = vd_basic(args)
    args = res18_la(args)
    args = slowfast_a4_basic(args)

    args['load_exp'] = 'vd_unsup_fx/dyn_clstr/vd_slowfast_a4'
    args['exp_id'] = 'vd_slowfast_a4_test_noimg'
    args['lr_boundaries'] = '1120011,1294998'
    args['plot_val'] = True
    args['pure_test'] = True
    args['plot_val_no_image'] = True
    return args


def vd_slowfast_a4_test_noimg_mre():
    args = {}

    args = basic_fix(args)
    args = slow_bs(args)
    args = vd_basic(args)
    args = res18_la(args)
    args = slowfast_a4_basic(args)

    args['load_exp'] = 'vd_unsup_fx/dyn_clstr/vd_slowfast_a4'
    args['exp_id'] = 'vd_slowfast_a4_test_noimg_mre'
    args['lr_boundaries'] = '1120011,1294998'
    args['plot_val'] = True
    args['pure_test'] = True
    args['plot_val_no_image'] = True
    args['test_no_frames'] = 10
    args['val_num_workers'] = 40
    return args
