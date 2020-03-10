def basic_rot(args):
    args['port'] = 27006
    args['db_name'] = 'vd_unsup_fx'
    args['col_name'] = 'rot'
    args['init_lr'] = 0.1
    return args


def bs128_rot(args):
    args['batch_size'] = 32
    args['test_batch_size'] = 32
    args['fre_filter'] = 50000
    args['fre_cache_filter'] = 5000
    args['fre_valid'] = 5000
    return args


def vd_basic_rot(args):
    args['image_dir'] = '/data5/chengxuz/Dataset/kinetics/comp_jpgs_extracted'
    args['val_image_dir'] = args['image_dir']
    return args


def rot_3dresnet():
    args = {}

    args = basic_rot(args)
    args = bs128_rot(args)
    args = vd_basic_rot(args)
    args['exp_id'] = 'rot_3dresnet'
    return args


def rot_3dresnet_re():
    args = {}

    args = basic_rot(args)
    args = bs128_rot(args)
    args = vd_basic_rot(args)
    args['exp_id'] = 'rot_3dresnet_re'
    args['rot_real_prep'] = True
    args['lr_boundaries'] = '25000,50000,75000'
    return args
