def basic_opn(args):
    args['port'] = 27006
    args['db_name'] = 'vd_unsup_fx'
    args['col_name'] = 'opn'
    return args


def bs128_opn(args):
    args['batch_size'] = 128
    args['test_batch_size'] = 64
    args['fre_filter'] = 50000
    args['fre_cache_filter'] = 5000
    args['fre_valid'] = 5000
    return args


def vd_basic_opn(args):
    args['image_dir'] = '/data5/chengxuz/Dataset/kinetics/comp_jpgs_extracted'
    args['val_image_dir'] = args['image_dir']
    return args


def opn_random():
    args = {}

    args = basic_opn(args)
    args = bs128_opn(args)
    args = vd_basic_opn(args)

    args['exp_id'] = 'opn_random'
    args['train_num_workers'] = 30
    args['lr_boundaries'] = '145011,215011'
    return args


def opn_random_224():
    args = {}

    args = basic_opn(args)
    args = bs128_opn(args)
    args = vd_basic_opn(args)

    args['opn_crop_size'] = 224
    args['exp_id'] = 'opn_random_224'
    args['train_num_workers'] = 30
    return args


def opn_random_224_sep():
    args = {}

    args = basic_opn(args)
    args = bs128_opn(args)
    args = vd_basic_opn(args)

    args['opn_crop_size'] = 224
    args['exp_id'] = 'opn_random_224_sep'
    args['train_num_workers'] = 30
    args['opn_transform'] = 'Sep'
    return args


def opn_random_sep_flow():
    args = {}

    args = basic_opn(args)
    args = bs128_opn(args)
    args = vd_basic_opn(args)

    args['exp_id'] = 'opn_random_sep_flow'
    args['train_num_workers'] = 30
    args['opn_transform'] = 'Sep'
    args['opn_flow_folder'] = '/data5/chengxuz/Dataset/kinetics/kinetics_flow25'
    args['lr_boundaries'] = '224998,269998'
    return args


def opn_random_224_sep_flow():
    args = {}

    args = basic_opn(args)
    args = bs128_opn(args)
    args = vd_basic_opn(args)

    args['opn_crop_size'] = 224
    args['exp_id'] = 'opn_random_224_sep_flow'
    args['train_num_workers'] = 30
    args['opn_transform'] = 'Sep'
    args['opn_flow_folder'] = '/data5/chengxuz/Dataset/kinetics/kinetics_flow25'
    args['lr_boundaries'] = '279998,344998'
    return args
