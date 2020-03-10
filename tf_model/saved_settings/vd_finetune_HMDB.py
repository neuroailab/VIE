def finetune_HMDB_bs128(args):
    args['batch_size'] = 128
    args['test_batch_size'] = 64
    args['test_no_frames'] = 5
    args['fre_filter'] = 10000
    args['fre_cache_filter'] = 5000
    args['fre_valid'] = 1000
    return args


def finetune_basics(args):
    # args['port'] = 27006
    args['port'] = 27007
    args['finetune_conv'] = True
    args['cache_dir'] = "/mnt/fs4/shetw/tfutils_cache"
    return args


def HMDB_basics(args):
    args['image_dir'] = '/data5/shetw/HMDB51/extracted_frames'
    args['val_image_dir'] = args['image_dir']
    args['metafile_root'] = '/data5/shetw/HMDB51/metafiles'
    args['dataset'] = 'HMDB51'
    args['train_len'] = 3570
    args['val_len'] = 1530
    args['train_prep'] = 'ColorJitter'
    args['num_classes'] = 51
    args['HMDB_sample'] = True
    return args


def vd_ctl_pool1():
    args = {}

    args = finetune_basics(args)
    args = HMDB_basics(args)
    args = finetune_HMDB_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_ctl"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/HMDB/vd_ctl_pool1"
    args['final_pooling'] = 1
    args["train_num_workers"] = 10
    args["lr_boundaries"] = '1750000,1850000'
    return args


def vd_tsrn_pool1():
    args = {}

    args = finetune_basics(args)
    args = HMDB_basics(args)
    args = finetune_HMDB_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_tsrn_f4_pret"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/HMDB/vd_tsrn_pool1"
    args['model_type'] = 'tsrn'
    args['trn_num_frames'] = 4
    args['get_all_layers'] = '9-time-avg'
    args["train_num_workers"] = 30
    args['final_pooling'] = 1

    args['lr_boundaries'] = '1755000,1855000'
    return args


def vd_slow_pool1():
    args = {}

    args = finetune_basics(args)
    args = HMDB_basics(args)
    args = finetune_HMDB_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_slow"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/HMDB/vd_slow_pool1"
    args['model_type'] = 'slow'
    args["train_num_workers"] = 40
    args['final_pooling'] = 1
    args["lr_boundaries"] = '1460000,1475000'
    return args