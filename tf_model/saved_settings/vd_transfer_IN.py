def trans_IN_bs128(args):
    args['batch_size'] = 128
    args['test_batch_size'] = 64
    args['fre_filter'] = 100090
    args['fre_cache_filter'] = 10009
    args['fre_valid'] = 10009
    return args


def trans_IN_bs64(args):
    args['batch_size'] = 64
    args['test_batch_size'] = 32
    args['fre_filter'] = 100090
    args['fre_cache_filter'] = 10009
    args['fre_valid'] = 10009
    return args


def trans_basics(args):
    args['port'] = 27006
    return args


def IN_basics(args):
    args['image_dir'] = '/mnt/fs1/Dataset/TFRecord_Imagenet_standard/image_label_full_widx'
    return args


def vd_from_scratch_trans_all():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_ctl"
    args['load_port'] = 27006
    args['resume'] = True
    args["save_exp"] = "vd_trans/IN/vd_from_scratch_trans_all"
    args["get_all_layers"] = '1,3,5,7,9'
    args["lr_boundaries"] = '370011,524998'
    return args


def vd_ctl_IR_trans_all():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_ctl_IR"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_ctl_IR_trans_all"
    args["get_all_layers"] = '1,3,5,7,9'
    args["lr_boundaries"] = '1840011,2192191'
    return args


def vd_ctl_trans():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_ctl"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_ctl_trans"
    args['lr_boundaries'] = '1491301,1891691'
    return args


def vd_ctl_trans_all():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_ctl"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_ctl_trans_all"
    args['lr_boundaries'] = '1491301,1891691'
    args["get_all_layers"] = '1,3,5,7,9'
    return args


def vd_ctl_p30_trans_all():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_ctl_p30"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_ctl_p30_trans_all"
    args["get_all_layers"] = '1,3,5,7,9'
    args['lr_boundaries'] = '1330011,1730011'
    return args


def vd_ctl_p70_trans_all():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_ctl_p70"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_ctl_p70_trans_all"
    args["get_all_layers"] = '1,3,5,7,9'
    args["lr_boundaries"] = '1370011,1600011'
    return args


def vd_ctl_big_bin_trans_all():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_ctl_big_bin"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_ctl_big_bin_trans_all"
    args['lr_boundaries'] = '1900011,2300011'
    args["get_all_layers"] = '1,3,5,7,9'
    return args


def vd_ctl_bin_trans_all():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_ctl_bin"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_ctl_bin_trans_all"
    args["get_all_layers"] = '1,3,5,7,9'
    args['lr_boundaries'] = '1811011,2100011'
    return args


def vd_sup_f1_trans():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_sup/ctl/vd_f1_ctl"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_sup_f1_trans"
    args["lr_boundaries"] = '790001,890001'
    return args


def vd_sup_f1_avg_trans():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_sup/ctl/vd_f1_ctl"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_sup_f1_avg_trans"
    args["get_all_layers"] = '9-avg'
    args["lr_boundaries"] = '790001,890001'
    return args


def vd_sup_f1_trans_all():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_sup/ctl/vd_f1_ctl"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_sup_f1_trans_all"
    args["get_all_layers"] = '1,3,5,7,9-avg'
    args["lr_boundaries"] = '790001,890001'
    return args


def vd_ctl_opn_trans():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/opn/opn_random"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_ctl_opn_trans_mlt"
    args["get_all_layers"] = '1,3,5,7,9'
    return args


def vd_ctl_opn_80_trans_all():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/opn/opn_random_sep_flow"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_ctl_opn_80_trans_all"
    args["get_all_layers"] = '1,3,5,7,9'
    args['lr_boundaries'] = '429998,670011'
    return args


def vd_ctl_opn_224_trans_all():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/opn/opn_random_224_sep_flow"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_ctl_opn_224_trans_all"
    args["get_all_layers"] = '1,3,5,7,9'
    return args


def vd_ctl_trans_alx():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_ctl"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_ctl_trans_alx"
    args['lr_boundaries'] = '1491301'
    args["train_crop"] = 'alexnet_crop_flip'
    return args


def vd_trn_pret_trans():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_trn_pret"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_trn_pret_trans_fx"
    return args


def vd_trn_f4_pret_trans():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_trn_f4_pret"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_trn_f4_pret_trans"
    args["lr_boundaries"] = '2440011,2650011'
    return args


def vd_trn_f4_pret_mlt_trans():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_trn_f4_pret"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_trn_f4_pret_mlt_trans"
    args["model_type"] = "trn_f4_tile"
    args["get_all_layers"] = 10
    args['lr_boundaries'] = '2360011'
    return args


def vd_sup_trn_f4_trans():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_sup/ctl/vd_sup_trn_f4_fx"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_sup_trn_f4_trans"
    return args


def vd_sup_trn_f4_avg_trans():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_sup/ctl/vd_sup_trn_f4_fx"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_sup_trn_f4_avg_trans"
    args["get_all_layers"] = "9-avg"
    args["lr_boundaries"] = '680011'
    return args


def vd_tsrn_slow_pret_trans():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_tsrn_slow_pret"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_tsrn_slow_pret_trans"
    args["lr_boundaries"] = '1901601'
    return args


def vd_tsrn_f4_pret_trans():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_tsrn_f4_pret"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_tsrn_f4_pret_trans"
    args["lr_boundaries"] = '1980011,2190011'
    return args


def vd_tsrn_f4_pret_trans_all():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    #args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_tsrn_f4_pret"
    #args["save_exp"] = "vd_trans/IN/vd_tsrn_f4_pret_trans_all"
    args["load_exp"] = "vd_trans/IN/vd_tsrn_f4_pret_trans_all"
    args["save_exp"] = "vd_trans/IN/vd_tsrn_f4_pret_trans_all_ct"
    args["load_step"] = 2181962

    args['load_port'] = 27006
    #args["lr_boundaries"] = '1980011,2190011'
    args["lr_boundaries"] = '1980011,2650011'
    args["get_all_layers"] = '1,3,5,7,9'
    return args


def vd_sup_tsrn_f4_pret_trans_all():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_sup/ctl/vd_sup_tsrn_f4"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_sup_tsrn_f4_pret_trans_all"
    args["get_all_layers"] = '1,3,5,7,9'
    args["lr_boundaries"] = '880011,1660011'
    return args


def vd_slow_trans():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_slow"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_slow_trans"
    args['model_type'] = 'slow'
    return args


def vd_slow_trans_all():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_slow"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_slow_trans_all"
    args['model_type'] = 'slow'
    args["get_all_layers"] = '1,3,5,7,9'
    args["num_tile"] = 4
    args["lr_boundaries"] = '1690011,1940011'
    return args


def vd_sup_slow_trans_all():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_sup/ctl/vd_sup_slow"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_sup_slow_trans_all"
    args['model_type'] = 'slow'
    args["get_all_layers"] = '1,3,5,7,9-avg'
    args["num_tile"] = 4
    args["lr_boundaries"] = '610011,1180011'
    return args


def vd_slowfast_a4_trans_all():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_slowfast_a4"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_slowfast_a4_trans_all_fx"
    args['model_type'] = 'slowfast_a4'
    args["get_all_layers"] = '1,3,5,7,9'
    args["num_tile"] = 16
    args["lr_boundaries"] = '1550011,1810011'
    return args


def vd_sup_slowfast_a4_trans_all():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_sup/ctl/vd_sup_slowfast_a4"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_sup_slowfast_a4_trans_all_fx"
    args['model_type'] = 'slowfast_a4'
    args["get_all_layers"] = '1,3,5,7,9-avg'
    args["num_tile"] = 16
    args["lr_boundaries"] = '530011,790011'
    return args


def vd_slow_single_trans_all():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_slow"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_slow_single_trans_all"
    args['model_type'] = 'slowsingle'
    args["get_all_layers"] = '5,7,9'
    args["num_tile"] = 4
    #args["from_ckpt"] = '/mnt/fs3/chengxuz/vd_relat/slow_single_model/model'
    args['lr_boundaries'] = '230011,570011'
    return args


def vd_slowfast_single_trans_all():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_slowfast_a4"
    args['load_port'] = 27006
    args["save_exp"] = "vd_trans/IN/vd_slowfast_single_trans_all"
    args['model_type'] = 'slowfastsingle'
    args["get_all_layers"] = '7,9'
    args["num_tile"] = 16
    #args["from_ckpt"] = '/mnt/fs3/chengxuz/vd_relat/slowfast_single_model/model'
    args['lr_boundaries'] = '160011,460011'
    return args


def vd_3dresnet_trans_all():
    args = {}

    args = trans_basics(args)
    args = IN_basics(args)
    args = trans_IN_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_3dresnet"
    args['load_port'] = 27007
    args["save_exp"] = "vd_trans/IN/vd_3dresnet_trans_all"
    args['port'] = 27007
    args['model_type'] = '3dresnet'
    args["get_all_layers"] = '5,7,9'
    args["num_tile"] = 16
    args["train_crop"] = 'outshape_112'
    args['lr_boundaries'] = '1850011,2500011'
    return args
