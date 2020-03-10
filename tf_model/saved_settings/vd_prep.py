from basics import basic, bs128
from vd_single_frame import vd_basic


def vd_ctl_IR_rdsz():
    args = {}

    args = basic(args)
    args = bs128(args)
    args = vd_basic(args)

    args['exp_id'] = 'vd_ctl_IR_rdsz'
    args['task'] = 'IR'
    args['train_prep'] = 'RandomSized'
    return args


def vd_ctl_IR_color():
    args = {}

    args = basic(args)
    args = bs128(args)
    args = vd_basic(args)

    args['exp_id'] = 'vd_ctl_IR_color'
    args['task'] = 'IR'
    args['train_prep'] = 'ColorJitter'
    args['lr_boundaries'] = '690011,1070011'
    return args
