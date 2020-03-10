from utils import DATA_LEN_KINETICS_400, VAL_DATA_LEN_KINETICS_400


def trans_KN_bs128(args):
    args['batch_size'] = 128
    args['test_batch_size'] = 64
    args['test_no_frames'] = 5
    args['fre_filter'] = 50000
    args['fre_cache_filter'] = 5000
    args['fre_valid'] = 5000
    return args


def trans_basics(args):
    args['port'] = 27006
    return args


def KN_basics(args):
    args['image_dir'] = '/data5/chengxuz/Dataset/kinetics/comp_jpgs_extracted'
    args['val_image_dir'] = args['image_dir']
    args['metafile_root'] = '/mnt/fs3/chengxuz/kinetics/pt_meta'
    args['dataset'] = 'kinetics'
    args['train_len'] = DATA_LEN_KINETICS_400
    args['val_len'] = VAL_DATA_LEN_KINETICS_400
    args['train_prep'] = 'ColorJitter'
    args['num_classes'] = 400
    return args


def finetune_UCF_bs128(args):
    args['batch_size'] = 128
    args['test_batch_size'] = 64
    args['test_no_frames'] = 5
    args['fre_filter'] = 10000
    args['fre_cache_filter'] = 2000
    args['fre_valid'] = 200
    return args


def finetune_basics(args):
    # args['port'] = 27006
    args['port'] = 27007
    args['finetune_conv'] = True
    return args


def UCF_basics(args):
    args['image_dir'] = '/data5/shetw/UCF101/extracted_frames'
    args['val_image_dir'] = args['image_dir']
    args['metafile_root'] = '/data5/shetw/UCF101/metafiles'
    args['dataset'] = 'UCF101'
    args['train_len'] = 9537
    args['val_len'] = 3783
    args['train_prep'] = 'ColorJitter'
    args['num_classes'] = 101
    return args


def vd_sup_ctl_UCF():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_sup/ctl/vd_f1_ctl"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_sup_ctl"
    args["train_num_workers"] = 5
    args["lr_boundaries"] = '725999,750000'
    return args


def vd_sup_ctl_UCF_wd5():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_sup/ctl/vd_f1_ctl"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_sup_ctl_wd5"
    args["train_num_workers"] = 5
    args["weight_decay"] = 5e-4
    args["lr_boundaries"] = '725999,750000'
    return args



def vd_from_scratch_trans_all():
    args = {}

    args = trans_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_ctl"
    args['load_port'] = 27006
    args['resume'] = True
    args["save_exp"] = "vd_trans/KN/vd_from_scratch_trans_all"
    args["get_all_layers"] = '1,3,5,7,9'
    args["train_num_workers"] = 40
    args["lr_boundaries"] = '409998,664998'
    return args


def vd_ctl_trans():
    args = {}

    args = trans_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_ctl"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/KN/vd_ctl_trans"
    args["train_num_workers"] = 40
    args["lr_boundaries"] = '1610099,1894998'
    return args


def vd_ctl_trans_all():
    args = {}

    args = trans_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_ctl"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/KN/vd_ctl_trans_all"
    args["train_num_workers"] = 40
    args["lr_boundaries"] = '1610099,1894998'
    args["get_all_layers"] = '1,3,5,7,9'
    return args


def vd_ctl_p30_trans_all():
    args = {}

    args = trans_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_ctl_p30"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/KN/vd_ctl_p30_trans_all"
    args["train_num_workers"] = 40
    args["get_all_layers"] = '1,3,5,7,9'
    args["lr_boundaries"] = '1444998,2079998'
    return args


def vd_ctl_p70_trans_all():
    args = {}

    args = trans_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_ctl_p70"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/KN/vd_ctl_p70_trans_all"
    args["train_num_workers"] = 40
    args["get_all_layers"] = '1,3,5,7,9'
    args["lr_boundaries"] = '1409998,1804998'
    return args


def vd_ctl_IR_trans_all():
    args = {}


    args = trans_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_ctl_IR"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/KN/vd_ctl_IR_trans_all"
    args["train_num_workers"] = 40
    args["lr_boundaries"] = '1740011,2120011'
    args["get_all_layers"] = '1,3,5,7,9'
    return args


def vd_ctl_bin_trans_all():
    args = {}

    args = trans_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_ctl_bin"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/KN/vd_ctl_bin_trans_all"
    args["get_all_layers"] = '1,3,5,7,9'
    args['lr_boundaries'] = '1811011,2050011'
    args["train_num_workers"] = 40
    return args


def vd_ctl_big_bin_trans_all():
    args = {}

    args = trans_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_ctl_big_bin"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/KN/vd_ctl_big_bin_trans_all"
    args['lr_boundaries'] = '1900011,2100011'
    args["get_all_layers"] = '1,3,5,7,9'
    args["train_num_workers"] = 40
    return args


def vd_slow_trans():
    args = {}

    args = trans_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_slow"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/KN/vd_slow_trans"
    args['model_type'] = 'slow'
    args["train_num_workers"] = 40
    args["lr_boundaries"] = '1539998,1630001'
    return args


def vd_slow_trans_all():
    args = {}

    args = trans_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    #args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_slow"
    #args["save_exp"] = "vd_trans/KN/vd_slow_trans_all"
    args["load_exp"] = "vd_trans/KN/vd_slow_trans_all"
    args['load_step'] = 1600000
    args["save_exp"] = "vd_trans/KN/vd_slow_trans_all_ct"

    args['load_port'] = 27006
    args['model_type'] = 'slow'
    args["train_num_workers"] = 40
    args["lr_boundaries"] = '1539998,1714998'
    args["get_all_layers"] = '1,3,5,7,9'
    return args


def vd_slow_single_trans_all():
    args = {}

    args = trans_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_slow"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/KN/vd_slow_single_trans_all"
    args['model_type'] = 'slowsingle_avg'
    args["get_all_layers"] = '5,7,9'
    args["train_num_workers"] = 40
    #args["from_ckpt"] = '/mnt/fs3/chengxuz/vd_relat/slow_single_model/model'
    args["test_batch_size"] = 32
    args["lr_boundaries"] = '104998,244998'
    return args


def vd_slowfast_a4_trans():
    args = {}

    args = trans_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_slowfast_a4"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/KN/vd_slowfast_a4_trans"
    args['model_type'] = 'slowfast_a4'
    args["train_num_workers"] = 40
    return args


def vd_slowfast_a4_trans_all():
    args = {}

    args = trans_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_slowfast_a4"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/KN/vd_slowfast_a4_trans_all"
    args['model_type'] = 'slowfast_a4'
    args["train_num_workers"] = 100
    args["get_all_layers"] = '1,3,5,7,9'
    args["lr_boundaries"] = '1469998,1594998'
    return args


def vd_slowfast_single_trans_all():
    args = {}

    args = trans_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_slowfast_a4"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/KN/vd_slowfast_single_trans_all"
    args['model_type'] = 'slowfastsingle_avg'
    args["get_all_layers"] = '5,7,9'
    args["train_num_workers"] = 100
    #args["from_ckpt"] = '/mnt/fs3/chengxuz/vd_relat/slowfast_single_model/model'
    args["test_batch_size"] = 32
    args["lr_boundaries"] = '134998,259998'
    return args


def vd_tsrn_slow_pret_trans():
    args = {}

    args = trans_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_tsrn_slow_pret"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/KN/vd_tsrn_slow_pret_trans"
    args['model_type'] = 'tsrn_slow'
    args['get_all_layers'] = '9-time-avg'
    args["train_num_workers"] = 40
    args["lr_boundaries"] = '1829998'
    return args


def vd_tsrn_f4_pret_trans():
    args = {}

    args = trans_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_tsrn_f4_pret"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/KN/vd_tsrn_f4_pret_trans"
    args['model_type'] = 'tsrn'
    args['trn_num_frames'] = 4
    args['get_all_layers'] = '9-time-avg'
    args["train_num_workers"] = 40
    args['test_batch_size'] = 32
    return args


def vd_tsrn_f4_pret_trans_all():
    args = {}

    args = trans_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    #args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_tsrn_f4_pret"
    #args["save_exp"] = "vd_trans/KN/vd_tsrn_f4_pret_trans_all"
    args["load_exp"] = "vd_trans/KN/vd_tsrn_f4_pret_trans_all"
    args["save_exp"] = "vd_trans/KN/vd_tsrn_f4_pret_trans_all_ct"
    args["load_step"] = 2050000

    args['load_port'] = 27006
    args['model_type'] = 'tsrn'
    args['trn_num_frames'] = 4
    args['get_all_layers'] = '1-time-avg,3-time-avg,5-time-avg,7-time-avg,9-time-avg'
    args["train_num_workers"] = 40
    args['test_batch_size'] = 32
    #args['lr_boundaries'] = '1854998,2049998'
    args['lr_boundaries'] = '1854998'
    return args


def vd_ctl_slow_trans_all():
    args = {}

    args = trans_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_ctl"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/KN/vd_ctl_slow_trans_all"
    args['model_type'] = 'tsrn_slow'
    args['get_all_layers'] = '1-time-avg,3-time-avg,5-time-avg,7-time-avg,9-time-avg'
    args["train_num_workers"] = 40
    args["lr_boundaries"] = '1529998,1719998'
    return args


def vd_ctl_tsrn_f4_trans_all():
    args = {}

    args = trans_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_ctl"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/KN/vd_ctl_tsrn_f4_trans_all"
    args['model_type'] = 'tsrn'
    args['trn_num_frames'] = 4
    args['get_all_layers'] = '1-time-avg,3-time-avg,5-time-avg,7-time-avg,9-time-avg'
    args["train_num_workers"] = 40
    args['test_batch_size'] = 32
    args['lr_boundaries'] = '1524998,1699998'
    return args


def vd_ctl_tsrn_slowfast_a4_trans_all():
    args = {}

    args = trans_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_ctl"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/KN/vd_ctl_tsrn_slowfast_a4_trans_all"
    args['model_type'] = 'tsrn_slowfast_a4'
    args["train_num_workers"] = 100
    args['get_all_layers'] = '1-sftime-avg,3-sftime-avg,5-sftime-avg,7-sftime-avg,9-sftime-avg'
    args['test_batch_size'] = 32
    args['lr_boundaries'] = '1469998,1584999'
    return args


def vd_trn_pret_trans():
    args = {}

    args = trans_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_trn_pret"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/KN/vd_trn_pret_trans"
    args['model_type'] = 'trn'
    args['get_all_layers'] = '10'
    args["train_num_workers"] = 40
    args['test_batch_size'] = 32
    return args


def vd_ctl_opn_80_trans_all():
    args = {}

    args = trans_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/opn/opn_random_sep_flow"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/KN/vd_ctl_opn_80_trans_all"
    args["get_all_layers"] = '1,3,5,7,9'
    args["train_num_workers"] = 40
    args['lr_boundaries'] = '479998,829998'
    return args


def vd_ctl_opn_224_trans_all():
    args = {}

    args = trans_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/opn/opn_random_224_sep_flow"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/KN/vd_ctl_opn_224_trans_all"
    args["get_all_layers"] = '1,3,5,7,9'
    args["train_num_workers"] = 40
    return args


def vd_3dresnet_trans_all():
    args = {}

    args = trans_basics(args)
    args = KN_basics(args)
    args = trans_KN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_3dresnet"
    args['load_port'] = 27007
    args["save_exp"] = "vd_trans/KN/vd_3dresnet_trans_all"
    args["port"] = 27007
    args['model_type'] = '3dresnet'
    args['train_prep'] = 'ColorJitter_112'

    args["train_num_workers"] = 100
    args["get_all_layers"] = '5,7,9'
    args['lr_boundaries'] = '1750011,1924998'
    return args
