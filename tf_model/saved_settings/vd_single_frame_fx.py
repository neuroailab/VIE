from saved_settings.basics import basic_fix, bs128
from utils import DATA_LEN_KINETICS_400


def vd_basic(args):
    args['dataset'] = 'kinetics'
    args['data_len'] = DATA_LEN_KINETICS_400
    args['image_dir'] = '/data5/chengxuz/Dataset/kinetics/comp_jpgs_extracted'
    args['val_image_dir'] = args['image_dir']
    args['train_prep'] = 'ColorJitter'
    return args


def UCF_basic(args):
    args['data_len'] = 9537
    args['image_dir'] = '/data5/shetw/UCF101/extracted_frames'
    args['val_image_dir'] = args['image_dir']
    args['metafile_root'] = '/data5/shetw/UCF101/metafiles'
    args['dataset'] = 'UCF101'
    args['train_len'] = 9537
    args['val_len'] = 3783
    args['num_classes'] = 101
    args['train_prep'] = 'ColorJitter'
    return args


def vd_ctl_IR():
    args = {}

    args = basic_fix(args)
    args = bs128(args)
    args = vd_basic(args)

    args['exp_id'] = 'vd_ctl_IR'
    args['task'] = 'IR'
    args['lr_boundaries'] = '845011,1280011'
    return args


def vd_ctl_p30_IR():
    args = {}

    args = basic_fix(args)
    args = bs128(args)
    args = vd_basic(args)

    args['exp_id'] = 'vd_ctl_p30_IR'
    args['task'] = 'IR'
    args['part_vd'] = 0.3
    args['data_len'] = int(DATA_LEN_KINETICS_400 * 0.3)
    args['lr_boundaries'] = '865011,1080011'
    return args


def vd_ctl_p70_IR():
    args = {}

    args = basic_fix(args)
    args = bs128(args)
    args = vd_basic(args)

    args['exp_id'] = 'vd_ctl_p70_IR'
    args['task'] = 'IR'
    args['part_vd'] = 0.7
    args['data_len'] = int(DATA_LEN_KINETICS_400 * 0.7)
    args['lr_boundaries'] = '875011,1080011'
    return args


def res18_la(args):
    args['kmeans_k'] = '8000'
    args['instance_k'] = 512
    return args


def load_from_IR(args):
    args['load_exp'] = 'vd_unsup_fx/dyn_clstr/vd_ctl_IR'
    args['load_step'] = 50000
    return args


def vd_ctl():
    args = {}

    args = basic_fix(args)
    args = bs128(args)
    args = vd_basic(args)
    args = load_from_IR(args)
    args = res18_la(args)

    args['exp_id'] = 'vd_ctl'
    args['lr_boundaries'] = '1020011,1220011'
    return args


def res18_la_p30(args):
    args['kmeans_k'] = '2400'
    args['instance_k'] = 512
    return args


def load_from_IR_p30(args):
    args['load_exp'] = 'vd_unsup_fx/dyn_clstr/vd_ctl_p30_IR'
    args['load_step'] = 50000
    return args


def vd_ctl_p30():
    args = {}

    args = basic_fix(args)
    args = bs128(args)
    args = vd_basic(args)
    args = load_from_IR_p30(args)
    args = res18_la_p30(args)

    args['exp_id'] = 'vd_ctl_p30'
    args['part_vd'] = 0.3
    args['data_len'] = int(DATA_LEN_KINETICS_400 * 0.3)
    args['lr_boundaries'] = '860011,1060011'
    return args


def res18_la_p70(args):
    args['kmeans_k'] = '5600'
    args['instance_k'] = 512
    return args


def load_from_IR_p70(args):
    args['load_exp'] = 'vd_unsup_fx/dyn_clstr/vd_ctl_p70_IR'
    args['load_step'] = 50000
    return args


def vd_ctl_p70():
    args = {}

    args = basic_fix(args)
    args = bs128(args)
    args = vd_basic(args)
    args = load_from_IR_p70(args)
    args = res18_la_p70(args)

    args['exp_id'] = 'vd_ctl_p70'
    args['part_vd'] = 0.7
    args['data_len'] = int(DATA_LEN_KINETICS_400 * 0.7)
    args['lr_boundaries'] = '860011,1060011'
    return args


def res18_la_smK(args):
    args['kmeans_k'] = '4000'
    args['instance_k'] = 512
    return args


def vd_ctl_smK():
    args = {}

    args = basic_fix(args)
    args = bs128(args)
    args = vd_basic(args)
    args = load_from_IR(args)
    args = res18_la_smK(args)

    args['exp_id'] = 'vd_ctl_smK'
    args['lr_boundaries'] = '1025011,1220011'
    return args


def res18_la_bgN(args):
    args['kmeans_k'] = '8000'
    args['instance_k'] = 1024
    return args


def vd_ctl_bgN():
    args = {}

    args = basic_fix(args)
    args = bs128(args)
    args = vd_basic(args)
    args = load_from_IR(args)
    args = res18_la_bgN(args)

    args['exp_id'] = 'vd_ctl_bgN'
    args['lr_boundaries'] = '985011,1255011'
    return args
