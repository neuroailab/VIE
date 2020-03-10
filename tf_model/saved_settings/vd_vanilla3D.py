from basics import basic
from vd_single_frame import vd_basic_color
from vd_fast import fast_bs


def vd_vanilla3D_IR():
    args = {}

    args = basic(args)
    args = fast_bs(args)
    args = vd_basic_color(args)

    args['exp_id'] = 'vd_vanilla3D_IR'
    args['model_type'] = 'vanilla3D'
    args['task'] = 'IR'
    args['train_num_workers'] = 20
    args['val_num_workers'] = 10
    return args
