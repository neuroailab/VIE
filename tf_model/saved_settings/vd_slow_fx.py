from saved_settings.basics import basic_fix
from saved_settings.vd_single_frame_fx import vd_basic, res18_la


def slow_bs(args):
    args['batch_size'] = 64
    args['test_batch_size'] = 32
    args['test_no_frames'] = 5
    args['kNN_val'] = 10
    args['fre_filter'] = 50000
    args['fre_cache_filter'] = 5000
    args['fre_valid'] = 5000
    return args


def vd_slow_IR():
    args = {}

    args = basic_fix(args)
    args = slow_bs(args)
    args = vd_basic(args)

    args['exp_id'] = 'vd_slow_IR'
    args['model_type'] = 'slow'
    args['task'] = 'IR'
    args['lr_boundaries'] = '1280011,1480011'
    return args


def load_from_slow_IR(args):
    args['load_exp'] = 'vd_unsup_fx/dyn_clstr/vd_slow_IR'
    args['load_step'] = 50000
    return args


def vd_slow():
    args = {}

    args = basic_fix(args)
    args = slow_bs(args)
    args = vd_basic(args)
    args = load_from_slow_IR(args)
    args = res18_la(args)

    args['exp_id'] = 'vd_slow'
    args['model_type'] = 'slow'
    args['lr_boundaries'] = '1170011,1380011'
    return args


def vd_slow_test_noimg():
    args = {}

    args = basic_fix(args)
    args = slow_bs(args)
    args = vd_basic(args)
    args = res18_la(args)

    args['load_exp'] = 'vd_unsup_fx/dyn_clstr/vd_slow'
    args['exp_id'] = 'vd_slow_test_noimg'
    args['model_type'] = 'slow'
    args['lr_boundaries'] = '1170011,1380011'
    args['plot_val'] = True
    args['pure_test'] = True
    args['plot_val_no_image'] = True
    return args


def vd_slow_test_noimg_mre():
    args = {}

    args = basic_fix(args)
    args = slow_bs(args)
    args = vd_basic(args)
    args = res18_la(args)

    args['load_exp'] = 'vd_unsup_fx/dyn_clstr/vd_slow'
    args['exp_id'] = 'vd_slow_test_noimg_mre'
    args['model_type'] = 'slow'
    args['lr_boundaries'] = '1170011,1380011'
    args['plot_val'] = True
    args['pure_test'] = True
    args['plot_val_no_image'] = True
    args['test_no_frames'] = 10
    args['val_num_workers'] = 40
    return args
