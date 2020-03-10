def trans_KN_bs128(args):
    args['batch_size'] = 128
    args['test_batch_size'] = 64
    args['test_no_frames'] = 5
    args['fre_filter'] = 50000
    args['fre_cache_filter'] = 5000
    args['fre_valid'] = 5000
    return args


def test_basics(args):
    args['port'] = 27007
    args['pure_test'] = True
    args['cache_dir'] = '/mnt/fs4/shetw/tfutils_cache'
    return args


def KN_basics(args):
    args['image_dir'] = '/data5/chengxuz/Dataset/kinetics/comp_jpgs_extracted'
    args['val_image_dir'] = args['image_dir']
    args['metafile_root'] = '/mnt/fs3/chengxuz/kinetics/pt_meta'
    args['dataset'] = 'kinetics'
    args['train_len'] = 239888
    args['val_len'] = 19653
    args['train_prep'] = 'ColorJitter'
    args['num_classes'] = 400
    return args


def vd_ctl_test():
    args = {}

    args = test_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_trans/KN/vd_ctl_trans"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans_test/KN/vd_ctl"
    args["train_num_workers"] = 40
    # args["lr_boundaries"] = '1610099,1894998'
    return args


def vd_slow_test():
    args = {}

    args = test_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_trans/KN/vd_slow_trans"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans_test/KN/vd_slow"
    args['model_type'] = 'slow'
    args["train_num_workers"] = 40
    # args["lr_boundaries"] = '1539998,1630001'
    return args


def vd_slowfast_a4_test():
    args = {}

    args = test_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_trans/KN/vd_slowfast_a4_trans"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans_test/KN/vd_slowfast_a4"
    args['model_type'] = 'slowfast_a4'
    args["train_num_workers"] = 40
    return args


def vd_tsrn_f4_pret_test():
    args = {}

    args = test_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_trans/KN/vd_tsrn_f4_pret_trans"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans_test/KN/vd_tsrn_f4"
    args['model_type'] = 'tsrn'
    args['trn_num_frames'] = 4
    args['get_all_layers'] = '9-time-avg'
    args["train_num_workers"] = 40
    args['test_batch_size'] = 32
    return args


def vd_ctl_all_test():
    args = {}

    args = test_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_trans/KN/vd_ctl_trans_all"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans_test/KN/vd_ctl_all"
    args["train_num_workers"] = 40
    # args["lr_boundaries"] = '1610099,1894998'
    args["get_all_layers"] = '1,3,5,7,9'
    return args


def vd_slow_all_test():
    args = {}

    args = test_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    #args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_slow"
    #args["save_exp"] = "vd_trans/KN/vd_slow_trans_all"
    args["load_exp"] = "vd_trans/KN/vd_slow_trans_all_ct"
    args["save_exp"] = "vd_trans_test/KN/vd_slow_all"

    args['load_port'] = 27006
    args['model_type'] = 'slow'
    args["train_num_workers"] = 40
    #args["lr_boundaries"] = '1539998,1714998'
    args["get_all_layers"] = '1,3,5,7,9'
    return args


def vd_slowfast_a4_all_test():
    args = {}

    args = test_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_trans/KN/vd_slowfast_a4_trans_all"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans_test/KN/vd_slowfast_all"
    args['model_type'] = 'slowfast_a4'
    args["train_num_workers"] = 40
    args["get_all_layers"] = '1,3,5,7,9'
    #args["lr_boundaries"] = '1469998,1594998'
    return args


def vd_tsrn_f4_pret_all_test():
    args = {}

    args = test_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    #args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_tsrn_f4_pret"
    #args["save_exp"] = "vd_trans/KN/vd_tsrn_f4_pret_trans_all"
    args["load_exp"] = "vd_trans/KN/vd_tsrn_f4_pret_trans_all"
    args["save_exp"] = "vd_trans_test/KN/vd_tsrn_all"
    # args["load_step"] = 2050000

    args['load_port'] = 27006
    args['model_type'] = 'tsrn'
    args['trn_num_frames'] = 4
    args['get_all_layers'] = '1-time-avg,3-time-avg,5-time-avg,7-time-avg,9-time-avg'
    args["train_num_workers"] = 40
    args['test_batch_size'] = 32
    #args['lr_boundaries'] = '1854998,2049998'
    # args['lr_boundaries'] = '1854998'
    return args