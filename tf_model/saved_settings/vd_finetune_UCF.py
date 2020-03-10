def finetune_UCF_bs128(args):
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
    args["save_exp"] = "vd_finetune/UCF/vd_sup_ctl_tmp"
    args["train_num_workers"] = 10
    args["lr_boundaries"] = '1450000,1480000'
    return args


def vd_ctl_UCF_pool1():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_ctl"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_ctl_pool1"
    args['final_pooling'] = 1
    args["train_num_workers"] = 10
    args["lr_boundaries"] = '1450000,1490000'
    return args


def vd_ctl_UCF_pool1_scalecrop():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_ctl"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_ctl_pool1_sc"
    args['final_pooling'] = 1
    args['train_prep'] = 'MultiScaleCrop_224'
    args["train_num_workers"] = 10
    args["lr_boundaries"] = '1450000,1490000'
    return args


def vd_sup_ctl_UCF_dropout5():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_sup/ctl/vd_f1_ctl"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_sup_ctl_dropout5_3"
    args['dropout'] = 0.5
    args["train_num_workers"] = 10
    args["lr_boundaries"] = '730000'
    return args


def vd_sup_ctl_UCF_dropout9():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_sup/ctl/vd_f1_ctl"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_sup_ctl_dropout9_3"
    args['dropout'] = 0.9
    args["train_num_workers"] = 10
    args["lr_boundaries"] = '730000'
    return args


def vd_sup_ctl_UCF_lr60():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_sup/ctl/vd_f1_ctl"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_sup_ctl_lr60"
    args["train_num_workers"] = 10
    args['init_lr'] = 0.075
    args["lr_boundaries"] = '750000,800000'
    return args


def vd_sup_ctl_UCF_lr50():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_sup/ctl/vd_f1_ctl"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_sup_ctl_lr50_3"
    args["train_num_workers"] = 10
    args['init_lr'] = 0.05
    args["lr_boundaries"] = '750000,800000'
    return args


def vd_sup_ctl_UCF_lr40():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_sup/ctl/vd_f1_ctl"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_sup_ctl_lr40"
    args["train_num_workers"] = 10
    args['init_lr'] = 0.04
    args["lr_boundaries"] = '750000,800000'
    return args


def vd_sup_ctl_UCF_lr5():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_sup/ctl/vd_f1_ctl"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_sup_ctl_lr5"
    args["train_num_workers"] = 10
    args['init_lr'] = 0.005
    args["lr_boundaries"] = '710000'
    return args


def vd_sup_ctl_UCF_lr30():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_sup/ctl/vd_f1_ctl"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_sup_ctl_lr30"
    args["train_num_workers"] = 10
    args['init_lr'] = 0.03
    args["lr_boundaries"] = '710000,715000'
    return args


def vd_sup_ctl_UCF_wd5():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_sup/ctl/vd_f1_ctl"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_sup_ctl_wd5"
    args["train_num_workers"] = 10
    args["weight_decay"] = 5e-4
    args["lr_boundaries"] = '1450000,1480000'
    return args


def vd_sup_ctl_UCF_wd10():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_sup/ctl/vd_f1_ctl"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_sup_ctl_wd10"
    args["train_num_workers"] = 10
    args["weight_decay"] = 1e-3
    args["lr_boundaries"] = '1450000,1480000'
    return args


tsrn_load_param_dict_no_momentum = {'global_step': 'global_step',
        'resnet_model/batch_normalization/beta': 'resnet_model/batch_normalization/beta',
        'resnet_model/batch_normalization/gamma': 'resnet_model/batch_normalization/gamma',
        'resnet_model/batch_normalization/moving_mean': 'resnet_model/batch_normalization/moving_mean',
        'resnet_model/batch_normalization/moving_variance': 'resnet_model/batch_normalization/moving_variance',
        'resnet_model/batch_normalization_1/beta': 'resnet_model/batch_normalization_1/beta',
        'resnet_model/batch_normalization_1/gamma': 'resnet_model/batch_normalization_1/gamma',
        'resnet_model/batch_normalization_1/moving_mean': 'resnet_model/batch_normalization_1/moving_mean',
        'resnet_model/batch_normalization_1/moving_variance': 'resnet_model/batch_normalization_1/moving_variance',
        'resnet_model/batch_normalization_10/beta': 'resnet_model/batch_normalization_10/beta',
        'resnet_model/batch_normalization_10/gamma': 'resnet_model/batch_normalization_10/gamma',
        'resnet_model/batch_normalization_10/moving_mean': 'resnet_model/batch_normalization_10/moving_mean',
        'resnet_model/batch_normalization_10/moving_variance': 'resnet_model/batch_normalization_10/moving_variance',
        'resnet_model/batch_normalization_11/beta': 'resnet_model/batch_normalization_11/beta',
        'resnet_model/batch_normalization_11/gamma': 'resnet_model/batch_normalization_11/gamma',
        'resnet_model/batch_normalization_11/moving_mean': 'resnet_model/batch_normalization_11/moving_mean',
        'resnet_model/batch_normalization_11/moving_variance': 'resnet_model/batch_normalization_11/moving_variance',
        'resnet_model/batch_normalization_12/beta': 'resnet_model/batch_normalization_12/beta',
        'resnet_model/batch_normalization_12/gamma': 'resnet_model/batch_normalization_12/gamma',
        'resnet_model/batch_normalization_12/moving_mean': 'resnet_model/batch_normalization_12/moving_mean',
        'resnet_model/batch_normalization_12/moving_variance': 'resnet_model/batch_normalization_12/moving_variance',
        'resnet_model/batch_normalization_13/beta': 'resnet_model/batch_normalization_13/beta',
        'resnet_model/batch_normalization_13/gamma': 'resnet_model/batch_normalization_13/gamma',
        'resnet_model/batch_normalization_13/moving_mean': 'resnet_model/batch_normalization_13/moving_mean',
        'resnet_model/batch_normalization_13/moving_variance': 'resnet_model/batch_normalization_13/moving_variance',
        'resnet_model/batch_normalization_14/beta': 'resnet_model/batch_normalization_14/beta',
        'resnet_model/batch_normalization_14/gamma': 'resnet_model/batch_normalization_14/gamma',
        'resnet_model/batch_normalization_14/moving_mean': 'resnet_model/batch_normalization_14/moving_mean',
        'resnet_model/batch_normalization_14/moving_variance': 'resnet_model/batch_normalization_14/moving_variance',
        'resnet_model/batch_normalization_15/beta': 'resnet_model/batch_normalization_15/beta',
        'resnet_model/batch_normalization_15/gamma': 'resnet_model/batch_normalization_15/gamma',
        'resnet_model/batch_normalization_15/moving_mean': 'resnet_model/batch_normalization_15/moving_mean',
        'resnet_model/batch_normalization_15/moving_variance': 'resnet_model/batch_normalization_15/moving_variance',
        'resnet_model/batch_normalization_16/beta': 'resnet_model/batch_normalization_16/beta',
        'resnet_model/batch_normalization_16/gamma': 'resnet_model/batch_normalization_16/gamma',
        'resnet_model/batch_normalization_16/moving_mean': 'resnet_model/batch_normalization_16/moving_mean',
        'resnet_model/batch_normalization_16/moving_variance': 'resnet_model/batch_normalization_16/moving_variance',
        'resnet_model/batch_normalization_2/beta': 'resnet_model/batch_normalization_2/beta',
        'resnet_model/batch_normalization_2/gamma': 'resnet_model/batch_normalization_2/gamma',
        'resnet_model/batch_normalization_2/moving_mean': 'resnet_model/batch_normalization_2/moving_mean',
        'resnet_model/batch_normalization_2/moving_variance': 'resnet_model/batch_normalization_2/moving_variance',
        'resnet_model/batch_normalization_3/beta': 'resnet_model/batch_normalization_3/beta',
        'resnet_model/batch_normalization_3/gamma': 'resnet_model/batch_normalization_3/gamma',
        'resnet_model/batch_normalization_3/moving_mean': 'resnet_model/batch_normalization_3/moving_mean',
        'resnet_model/batch_normalization_3/moving_variance': 'resnet_model/batch_normalization_3/moving_variance',
        'resnet_model/batch_normalization_4/beta': 'resnet_model/batch_normalization_4/beta',
        'resnet_model/batch_normalization_4/gamma': 'resnet_model/batch_normalization_4/gamma',
        'resnet_model/batch_normalization_4/moving_mean': 'resnet_model/batch_normalization_4/moving_mean',
        'resnet_model/batch_normalization_4/moving_variance': 'resnet_model/batch_normalization_4/moving_variance',
        'resnet_model/batch_normalization_5/beta': 'resnet_model/batch_normalization_5/beta',
        'resnet_model/batch_normalization_5/gamma': 'resnet_model/batch_normalization_5/gamma',
        'resnet_model/batch_normalization_5/moving_mean': 'resnet_model/batch_normalization_5/moving_mean',
        'resnet_model/batch_normalization_5/moving_variance': 'resnet_model/batch_normalization_5/moving_variance',
        'resnet_model/batch_normalization_6/beta': 'resnet_model/batch_normalization_6/beta',
        'resnet_model/batch_normalization_6/gamma': 'resnet_model/batch_normalization_6/gamma',
        'resnet_model/batch_normalization_6/moving_mean': 'resnet_model/batch_normalization_6/moving_mean',
        'resnet_model/batch_normalization_6/moving_variance': 'resnet_model/batch_normalization_6/moving_variance',
        'resnet_model/batch_normalization_7/beta': 'resnet_model/batch_normalization_7/beta',
        'resnet_model/batch_normalization_7/gamma': 'resnet_model/batch_normalization_7/gamma',
        'resnet_model/batch_normalization_7/moving_mean': 'resnet_model/batch_normalization_7/moving_mean',
        'resnet_model/batch_normalization_7/moving_variance': 'resnet_model/batch_normalization_7/moving_variance',
        'resnet_model/batch_normalization_8/beta': 'resnet_model/batch_normalization_8/beta',
        'resnet_model/batch_normalization_8/gamma': 'resnet_model/batch_normalization_8/gamma',
        'resnet_model/batch_normalization_8/moving_mean': 'resnet_model/batch_normalization_8/moving_mean',
        'resnet_model/batch_normalization_8/moving_variance': 'resnet_model/batch_normalization_8/moving_variance',
        'resnet_model/batch_normalization_9/beta': 'resnet_model/batch_normalization_9/beta',
        'resnet_model/batch_normalization_9/gamma': 'resnet_model/batch_normalization_9/gamma',
        'resnet_model/batch_normalization_9/moving_mean': 'resnet_model/batch_normalization_9/moving_mean',
        'resnet_model/batch_normalization_9/moving_variance': 'resnet_model/batch_normalization_9/moving_variance',
        'resnet_model/conv2d/kernel': 'resnet_model/conv2d/kernel',
        'resnet_model/conv2d_1/kernel': 'resnet_model/conv2d_1/kernel',
        'resnet_model/conv2d_10/kernel': 'resnet_model/conv2d_10/kernel',
        'resnet_model/conv2d_11/kernel': 'resnet_model/conv2d_11/kernel',
        'resnet_model/conv2d_12/kernel': 'resnet_model/conv2d_12/kernel',
        'resnet_model/conv2d_13/kernel': 'resnet_model/conv2d_13/kernel',
        'resnet_model/conv2d_14/kernel': 'resnet_model/conv2d_14/kernel',
        'resnet_model/conv2d_15/kernel': 'resnet_model/conv2d_15/kernel',
        'resnet_model/conv2d_16/kernel': 'resnet_model/conv2d_16/kernel',
        'resnet_model/conv2d_17/kernel': 'resnet_model/conv2d_17/kernel',
        'resnet_model/conv2d_18/kernel': 'resnet_model/conv2d_18/kernel',
        'resnet_model/conv2d_19/kernel': 'resnet_model/conv2d_19/kernel',
        'resnet_model/conv2d_2/kernel': 'resnet_model/conv2d_2/kernel',
        'resnet_model/conv2d_20/kernel': 'resnet_model/conv2d_20/kernel',
        'resnet_model/conv2d_3/kernel': 'resnet_model/conv2d_3/kernel',
        'resnet_model/conv2d_4/kernel': 'resnet_model/conv2d_4/kernel',
        'resnet_model/conv2d_5/kernel': 'resnet_model/conv2d_5/kernel',
        'resnet_model/conv2d_6/kernel': 'resnet_model/conv2d_6/kernel',
        'resnet_model/conv2d_7/kernel': 'resnet_model/conv2d_7/kernel',
        'resnet_model/conv2d_8/kernel': 'resnet_model/conv2d_8/kernel',
        'resnet_model/conv2d_9/kernel': 'resnet_model/conv2d_9/kernel',
        'resnet_model/dense/bias': 'resnet_model/dense/bias',
        'resnet_model/dense/kernel': 'resnet_model/dense/kernel'}


def vd_sup_tsrn_UCF():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_sup/ctl/vd_sup_tsrn_f4"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_sup_tsrn_lr10_2"
    args["train_num_workers"] = 10
    # args['init_lr'] = 0.05
    args['lr_boundaries'] = '575000'
    return args


def vd_sup_tsrn_UCF_trans():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_sup/ctl/vd_sup_tsrn_f4"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_sup_tsrn_trans"
    args['finetune_conv'] = False
    args["train_num_workers"] = 10
    args['init_lr'] = 0.01
    args['lr_boundaries'] = '575000,620000'
    return args


def vd_sup_tsrn_UCF_adam():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_sup/ctl/vd_sup_tsrn_f4"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_sup_tsrn_adam"
    args["train_num_workers"] = 10
    args['optimizer'] = "Adam"
    args['init_lr'] = 0.001
    args['lr_boundaries'] = '575000'
    return args


def vd_tsrn_UCF():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_tsrn_f4_pret"
    args['load_port'] = 27006
    #args['load_exp'] = None
    args["save_exp"] = "vd_finetune/UCF/vd_tsrn_3"
    args['model_type'] = 'tsrn'
    args['trn_num_frames'] = 4
    args['get_all_layers'] = '9-time-avg'
    args['finetune_conv'] = True
    args["train_num_workers"] = 30
    args['lr_boundaries'] = '1710000,1750000'
    return args


def vd_tsrn_UCF_pool1():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_tsrn_f4_pret"
    args['load_port'] = 27006
    #args['load_exp'] = None
    args["save_exp"] = "vd_finetune/UCF/vd_tsrn_pool1_2"
    args['model_type'] = 'tsrn'
    args['trn_num_frames'] = 4
    args['get_all_layers'] = '9-time-avg'
    args['finetune_conv'] = True
    args["train_num_workers"] = 30

    args['final_pooling'] = 1

    args['lr_boundaries'] = '1710000,1750000'
    return args


def vd_tsrn_UCF_pool3():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_tsrn_f4_pret"
    args['load_port'] = 27006
    #args['load_exp'] = None
    args["save_exp"] = "vd_finetune/UCF/vd_tsrn_3_pool3_2"
    args['model_type'] = 'tsrn'
    args['trn_num_frames'] = 4
    args['get_all_layers'] = '9-time-avg'
    args['finetune_conv'] = True
    args["train_num_workers"] = 30

    args['final_pooling'] = 3

    args['lr_boundaries'] = '1710000,1750000'
    return args


def vd_tsrn_UCF_trans():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_tsrn_f4_pret"
    args['load_port'] = 27006
    #args['load_exp'] = None
    args["save_exp"] = "vd_finetune/UCF/vd_tsrn"
    args['finetune_conv'] = False
    args["train_num_workers"] = 10
    args['lr_boundaries'] = '1700000,1730000'
    return args


def vd_tsrn_UCF_trans_finetune():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_finetune/UCF/vd_tsrn"
    args['load_port'] = 27007
    args['load_step'] = 1760000
    args['save_exp'] = "vd_finetune/UCF/vd_tsrn_trans_finetune"
    args['finetune_conv'] = True
    args["train_num_workers"] = 10
    args['lr_boundaries'] = '1790000,1820000'
    return args


def vd_slow_UCF():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_slow"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_slow_3"
    args['model_type'] = 'slow'
    args["train_num_workers"] = 30
    args["lr_boundaries"] = '1540000,1605000'
    return args


def vd_slow_UCF_pool1():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_slow"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_slow_pool1"
    args['model_type'] = 'slow'
    args["train_num_workers"] = 40
    args['final_pooling'] = 1
    args["lr_boundaries"] = '1505000,1650000'
    return args


def vd_slow_all_UCF():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_slow"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_slow_all"
    args['model_type'] = 'slow'
    args["get_all_layers"] = '1,3,5,7,9'
    args["train_num_workers"] = 40
    # Note: Need to change during training
    args["lr_boundaries"] = '1539998,1630001'
    return args


slowfast_load_param_dict_no_momentum = {'global_step': 'global_step',
        'instance/all_labels': 'instance/all_labels',
        'instance/cluster_labels': 'instance/cluster_labels',
        'instance/memory_bank': 'instance/memory_bank',
        'resnet_model_fast/batch_normalization/beta': 'resnet_model_fast/batch_normalization/beta',
        'resnet_model_fast/batch_normalization/gamma': 'resnet_model_fast/batch_normalization/gamma',
        'resnet_model_fast/batch_normalization/moving_mean': 'resnet_model_fast/batch_normalization/moving_mean',
        'resnet_model_fast/batch_normalization/moving_variance': 'resnet_model_fast/batch_normalization/moving_variance',
        'resnet_model_fast/batch_normalization_1/beta': 'resnet_model_fast/batch_normalization_1/beta',
        'resnet_model_fast/batch_normalization_1/gamma': 'resnet_model_fast/batch_normalization_1/gamma',
        'resnet_model_fast/batch_normalization_1/moving_mean': 'resnet_model_fast/batch_normalization_1/moving_mean',
        'resnet_model_fast/batch_normalization_1/moving_variance': 'resnet_model_fast/batch_normalization_1/moving_variance',
        'resnet_model_fast/batch_normalization_10/beta': 'resnet_model_fast/batch_normalization_10/beta',
        'resnet_model_fast/batch_normalization_10/gamma': 'resnet_model_fast/batch_normalization_10/gamma',
        'resnet_model_fast/batch_normalization_10/moving_mean': 'resnet_model_fast/batch_normalization_10/moving_mean',
        'resnet_model_fast/batch_normalization_10/moving_variance': 'resnet_model_fast/batch_normalization_10/moving_variance',
        'resnet_model_fast/batch_normalization_11/beta': 'resnet_model_fast/batch_normalization_11/beta',
        'resnet_model_fast/batch_normalization_11/gamma': 'resnet_model_fast/batch_normalization_11/gamma',
        'resnet_model_fast/batch_normalization_11/moving_mean': 'resnet_model_fast/batch_normalization_11/moving_mean',
        'resnet_model_fast/batch_normalization_11/moving_variance': 'resnet_model_fast/batch_normalization_11/moving_variance',
        'resnet_model_fast/batch_normalization_12/beta': 'resnet_model_fast/batch_normalization_12/beta',
        'resnet_model_fast/batch_normalization_12/gamma': 'resnet_model_fast/batch_normalization_12/gamma',
        'resnet_model_fast/batch_normalization_12/moving_mean': 'resnet_model_fast/batch_normalization_12/moving_mean',
        'resnet_model_fast/batch_normalization_12/moving_variance': 'resnet_model_fast/batch_normalization_12/moving_variance',
        'resnet_model_fast/batch_normalization_13/beta': 'resnet_model_fast/batch_normalization_13/beta',
        'resnet_model_fast/batch_normalization_13/gamma': 'resnet_model_fast/batch_normalization_13/gamma',
        'resnet_model_fast/batch_normalization_13/moving_mean': 'resnet_model_fast/batch_normalization_13/moving_mean',
        'resnet_model_fast/batch_normalization_13/moving_variance': 'resnet_model_fast/batch_normalization_13/moving_variance',
        'resnet_model_fast/batch_normalization_14/beta': 'resnet_model_fast/batch_normalization_14/beta',
        'resnet_model_fast/batch_normalization_14/gamma': 'resnet_model_fast/batch_normalization_14/gamma',
        'resnet_model_fast/batch_normalization_14/moving_mean': 'resnet_model_fast/batch_normalization_14/moving_mean',
        'resnet_model_fast/batch_normalization_14/moving_variance': 'resnet_model_fast/batch_normalization_14/moving_variance',
        'resnet_model_fast/batch_normalization_15/beta': 'resnet_model_fast/batch_normalization_15/beta',
        'resnet_model_fast/batch_normalization_15/gamma': 'resnet_model_fast/batch_normalization_15/gamma',
        'resnet_model_fast/batch_normalization_15/moving_mean': 'resnet_model_fast/batch_normalization_15/moving_mean',
        'resnet_model_fast/batch_normalization_15/moving_variance': 'resnet_model_fast/batch_normalization_15/moving_variance',
        'resnet_model_fast/batch_normalization_16/beta': 'resnet_model_fast/batch_normalization_16/beta',
        'resnet_model_fast/batch_normalization_16/gamma': 'resnet_model_fast/batch_normalization_16/gamma',
        'resnet_model_fast/batch_normalization_16/moving_mean': 'resnet_model_fast/batch_normalization_16/moving_mean',
        'resnet_model_fast/batch_normalization_16/moving_variance': 'resnet_model_fast/batch_normalization_16/moving_variance',
        'resnet_model_fast/batch_normalization_2/beta': 'resnet_model_fast/batch_normalization_2/beta',
        'resnet_model_fast/batch_normalization_2/gamma': 'resnet_model_fast/batch_normalization_2/gamma',
        'resnet_model_fast/batch_normalization_2/moving_mean': 'resnet_model_fast/batch_normalization_2/moving_mean',
        'resnet_model_fast/batch_normalization_2/moving_variance': 'resnet_model_fast/batch_normalization_2/moving_variance',
        'resnet_model_fast/batch_normalization_3/beta': 'resnet_model_fast/batch_normalization_3/beta',
        'resnet_model_fast/batch_normalization_3/gamma': 'resnet_model_fast/batch_normalization_3/gamma',
        'resnet_model_fast/batch_normalization_3/moving_mean': 'resnet_model_fast/batch_normalization_3/moving_mean',
        'resnet_model_fast/batch_normalization_3/moving_variance': 'resnet_model_fast/batch_normalization_3/moving_variance',
        'resnet_model_fast/batch_normalization_4/beta': 'resnet_model_fast/batch_normalization_4/beta',
        'resnet_model_fast/batch_normalization_4/gamma': 'resnet_model_fast/batch_normalization_4/gamma',
        'resnet_model_fast/batch_normalization_4/moving_mean': 'resnet_model_fast/batch_normalization_4/moving_mean',
        'resnet_model_fast/batch_normalization_4/moving_variance': 'resnet_model_fast/batch_normalization_4/moving_variance',
        'resnet_model_fast/batch_normalization_5/beta': 'resnet_model_fast/batch_normalization_5/beta',
        'resnet_model_fast/batch_normalization_5/gamma': 'resnet_model_fast/batch_normalization_5/gamma',
        'resnet_model_fast/batch_normalization_5/moving_mean': 'resnet_model_fast/batch_normalization_5/moving_mean',
        'resnet_model_fast/batch_normalization_5/moving_variance': 'resnet_model_fast/batch_normalization_5/moving_variance',
        'resnet_model_fast/batch_normalization_6/beta': 'resnet_model_fast/batch_normalization_6/beta',
        'resnet_model_fast/batch_normalization_6/gamma': 'resnet_model_fast/batch_normalization_6/gamma',
        'resnet_model_fast/batch_normalization_6/moving_mean': 'resnet_model_fast/batch_normalization_6/moving_mean',
        'resnet_model_fast/batch_normalization_6/moving_variance': 'resnet_model_fast/batch_normalization_6/moving_variance',
        'resnet_model_fast/batch_normalization_7/beta': 'resnet_model_fast/batch_normalization_7/beta',
        'resnet_model_fast/batch_normalization_7/gamma': 'resnet_model_fast/batch_normalization_7/gamma',
        'resnet_model_fast/batch_normalization_7/moving_mean': 'resnet_model_fast/batch_normalization_7/moving_mean',
        'resnet_model_fast/batch_normalization_7/moving_variance': 'resnet_model_fast/batch_normalization_7/moving_variance',
        'resnet_model_fast/batch_normalization_8/beta': 'resnet_model_fast/batch_normalization_8/beta',
        'resnet_model_fast/batch_normalization_8/gamma': 'resnet_model_fast/batch_normalization_8/gamma',
        'resnet_model_fast/batch_normalization_8/moving_mean': 'resnet_model_fast/batch_normalization_8/moving_mean',
        'resnet_model_fast/batch_normalization_8/moving_variance': 'resnet_model_fast/batch_normalization_8/moving_variance',
        'resnet_model_fast/batch_normalization_9/beta': 'resnet_model_fast/batch_normalization_9/beta',
        'resnet_model_fast/batch_normalization_9/gamma': 'resnet_model_fast/batch_normalization_9/gamma',
        'resnet_model_fast/batch_normalization_9/moving_mean': 'resnet_model_fast/batch_normalization_9/moving_mean',
        'resnet_model_fast/batch_normalization_9/moving_variance': 'resnet_model_fast/batch_normalization_9/moving_variance',
        'resnet_model_fast/conv3d/kernel': 'resnet_model_fast/conv3d/kernel',
        'resnet_model_fast/conv3d_1/kernel': 'resnet_model_fast/conv3d_1/kernel',
        'resnet_model_fast/conv3d_10/kernel': 'resnet_model_fast/conv3d_10/kernel',
        'resnet_model_fast/conv3d_11/kernel': 'resnet_model_fast/conv3d_11/kernel',
        'resnet_model_fast/conv3d_12/kernel': 'resnet_model_fast/conv3d_12/kernel',
        'resnet_model_fast/conv3d_13/kernel': 'resnet_model_fast/conv3d_13/kernel',
        'resnet_model_fast/conv3d_14/kernel': 'resnet_model_fast/conv3d_14/kernel',
        'resnet_model_fast/conv3d_15/kernel': 'resnet_model_fast/conv3d_15/kernel',
        'resnet_model_fast/conv3d_16/kernel': 'resnet_model_fast/conv3d_16/kernel',
        'resnet_model_fast/conv3d_17/kernel': 'resnet_model_fast/conv3d_17/kernel',
        'resnet_model_fast/conv3d_18/kernel': 'resnet_model_fast/conv3d_18/kernel',
        'resnet_model_fast/conv3d_19/kernel': 'resnet_model_fast/conv3d_19/kernel',
        'resnet_model_fast/conv3d_2/kernel': 'resnet_model_fast/conv3d_2/kernel',
        'resnet_model_fast/conv3d_20/kernel': 'resnet_model_fast/conv3d_20/kernel',
        'resnet_model_fast/conv3d_3/kernel': 'resnet_model_fast/conv3d_3/kernel',
        'resnet_model_fast/conv3d_4/kernel': 'resnet_model_fast/conv3d_4/kernel',
        'resnet_model_fast/conv3d_5/kernel': 'resnet_model_fast/conv3d_5/kernel',
        'resnet_model_fast/conv3d_6/kernel': 'resnet_model_fast/conv3d_6/kernel',
        'resnet_model_fast/conv3d_7/kernel': 'resnet_model_fast/conv3d_7/kernel',
        'resnet_model_fast/conv3d_8/kernel': 'resnet_model_fast/conv3d_8/kernel',
        'resnet_model_fast/conv3d_9/kernel': 'resnet_model_fast/conv3d_9/kernel',
        'resnet_model_slow/batch_normalization/beta': 'resnet_model_slow/batch_normalization/beta',
        'resnet_model_slow/batch_normalization/gamma': 'resnet_model_slow/batch_normalization/gamma',
        'resnet_model_slow/batch_normalization/moving_mean': 'resnet_model_slow/batch_normalization/moving_mean',
        'resnet_model_slow/batch_normalization/moving_variance': 'resnet_model_slow/batch_normalization/moving_variance',
        'resnet_model_slow/batch_normalization_1/beta': 'resnet_model_slow/batch_normalization_1/beta',
        'resnet_model_slow/batch_normalization_1/gamma': 'resnet_model_slow/batch_normalization_1/gamma',
        'resnet_model_slow/batch_normalization_1/moving_mean': 'resnet_model_slow/batch_normalization_1/moving_mean',
        'resnet_model_slow/batch_normalization_1/moving_variance': 'resnet_model_slow/batch_normalization_1/moving_variance',
        'resnet_model_slow/batch_normalization_10/beta': 'resnet_model_slow/batch_normalization_10/beta',
        'resnet_model_slow/batch_normalization_10/gamma': 'resnet_model_slow/batch_normalization_10/gamma',
        'resnet_model_slow/batch_normalization_10/moving_mean': 'resnet_model_slow/batch_normalization_10/moving_mean',
        'resnet_model_slow/batch_normalization_10/moving_variance': 'resnet_model_slow/batch_normalization_10/moving_variance',
        'resnet_model_slow/batch_normalization_11/beta': 'resnet_model_slow/batch_normalization_11/beta',
        'resnet_model_slow/batch_normalization_11/gamma': 'resnet_model_slow/batch_normalization_11/gamma',
        'resnet_model_slow/batch_normalization_11/moving_mean': 'resnet_model_slow/batch_normalization_11/moving_mean',
        'resnet_model_slow/batch_normalization_11/moving_variance': 'resnet_model_slow/batch_normalization_11/moving_variance',
        'resnet_model_slow/batch_normalization_12/beta': 'resnet_model_slow/batch_normalization_12/beta',
        'resnet_model_slow/batch_normalization_12/gamma': 'resnet_model_slow/batch_normalization_12/gamma',
        'resnet_model_slow/batch_normalization_12/moving_mean': 'resnet_model_slow/batch_normalization_12/moving_mean',
        'resnet_model_slow/batch_normalization_12/moving_variance': 'resnet_model_slow/batch_normalization_12/moving_variance',
        'resnet_model_slow/batch_normalization_13/beta': 'resnet_model_slow/batch_normalization_13/beta',
        'resnet_model_slow/batch_normalization_13/gamma': 'resnet_model_slow/batch_normalization_13/gamma',
        'resnet_model_slow/batch_normalization_13/moving_mean': 'resnet_model_slow/batch_normalization_13/moving_mean',
        'resnet_model_slow/batch_normalization_13/moving_variance': 'resnet_model_slow/batch_normalization_13/moving_variance',
        'resnet_model_slow/batch_normalization_14/beta': 'resnet_model_slow/batch_normalization_14/beta',
        'resnet_model_slow/batch_normalization_14/gamma': 'resnet_model_slow/batch_normalization_14/gamma',
        'resnet_model_slow/batch_normalization_14/moving_mean': 'resnet_model_slow/batch_normalization_14/moving_mean',
        'resnet_model_slow/batch_normalization_14/moving_variance': 'resnet_model_slow/batch_normalization_14/moving_variance',
        'resnet_model_slow/batch_normalization_15/beta': 'resnet_model_slow/batch_normalization_15/beta',
        'resnet_model_slow/batch_normalization_15/gamma': 'resnet_model_slow/batch_normalization_15/gamma',
        'resnet_model_slow/batch_normalization_15/moving_mean': 'resnet_model_slow/batch_normalization_15/moving_mean',
        'resnet_model_slow/batch_normalization_15/moving_variance': 'resnet_model_slow/batch_normalization_15/moving_variance',
        'resnet_model_slow/batch_normalization_16/beta': 'resnet_model_slow/batch_normalization_16/beta',
        'resnet_model_slow/batch_normalization_16/gamma': 'resnet_model_slow/batch_normalization_16/gamma',
        'resnet_model_slow/batch_normalization_16/moving_mean': 'resnet_model_slow/batch_normalization_16/moving_mean',
        'resnet_model_slow/batch_normalization_16/moving_variance': 'resnet_model_slow/batch_normalization_16/moving_variance',
        'resnet_model_slow/batch_normalization_2/beta': 'resnet_model_slow/batch_normalization_2/beta',
        'resnet_model_slow/batch_normalization_2/gamma': 'resnet_model_slow/batch_normalization_2/gamma',
        'resnet_model_slow/batch_normalization_2/moving_mean': 'resnet_model_slow/batch_normalization_2/moving_mean',
        'resnet_model_slow/batch_normalization_2/moving_variance': 'resnet_model_slow/batch_normalization_2/moving_variance',
        'resnet_model_slow/batch_normalization_3/beta': 'resnet_model_slow/batch_normalization_3/beta',
        'resnet_model_slow/batch_normalization_3/gamma': 'resnet_model_slow/batch_normalization_3/gamma',
        'resnet_model_slow/batch_normalization_3/moving_mean': 'resnet_model_slow/batch_normalization_3/moving_mean',
        'resnet_model_slow/batch_normalization_3/moving_variance': 'resnet_model_slow/batch_normalization_3/moving_variance',
        'resnet_model_slow/batch_normalization_4/beta': 'resnet_model_slow/batch_normalization_4/beta',
        'resnet_model_slow/batch_normalization_4/gamma': 'resnet_model_slow/batch_normalization_4/gamma',
        'resnet_model_slow/batch_normalization_4/moving_mean': 'resnet_model_slow/batch_normalization_4/moving_mean',
        'resnet_model_slow/batch_normalization_4/moving_variance': 'resnet_model_slow/batch_normalization_4/moving_variance',
        'resnet_model_slow/batch_normalization_5/beta': 'resnet_model_slow/batch_normalization_5/beta',
        'resnet_model_slow/batch_normalization_5/gamma': 'resnet_model_slow/batch_normalization_5/gamma',
        'resnet_model_slow/batch_normalization_5/moving_mean': 'resnet_model_slow/batch_normalization_5/moving_mean',
        'resnet_model_slow/batch_normalization_5/moving_variance': 'resnet_model_slow/batch_normalization_5/moving_variance',
        'resnet_model_slow/batch_normalization_6/beta': 'resnet_model_slow/batch_normalization_6/beta',
        'resnet_model_slow/batch_normalization_6/gamma': 'resnet_model_slow/batch_normalization_6/gamma',
        'resnet_model_slow/batch_normalization_6/moving_mean': 'resnet_model_slow/batch_normalization_6/moving_mean',
        'resnet_model_slow/batch_normalization_6/moving_variance': 'resnet_model_slow/batch_normalization_6/moving_variance',
        'resnet_model_slow/batch_normalization_7/beta': 'resnet_model_slow/batch_normalization_7/beta',
        'resnet_model_slow/batch_normalization_7/gamma': 'resnet_model_slow/batch_normalization_7/gamma',
        'resnet_model_slow/batch_normalization_7/moving_mean': 'resnet_model_slow/batch_normalization_7/moving_mean',
        'resnet_model_slow/batch_normalization_7/moving_variance': 'resnet_model_slow/batch_normalization_7/moving_variance',
        'resnet_model_slow/batch_normalization_8/beta': 'resnet_model_slow/batch_normalization_8/beta',
        'resnet_model_slow/batch_normalization_8/gamma': 'resnet_model_slow/batch_normalization_8/gamma',
        'resnet_model_slow/batch_normalization_8/moving_mean': 'resnet_model_slow/batch_normalization_8/moving_mean',
        'resnet_model_slow/batch_normalization_8/moving_variance': 'resnet_model_slow/batch_normalization_8/moving_variance',
        'resnet_model_slow/batch_normalization_9/beta': 'resnet_model_slow/batch_normalization_9/beta',
        'resnet_model_slow/batch_normalization_9/gamma': 'resnet_model_slow/batch_normalization_9/gamma',
        'resnet_model_slow/batch_normalization_9/moving_mean': 'resnet_model_slow/batch_normalization_9/moving_mean',
        'resnet_model_slow/batch_normalization_9/moving_variance': 'resnet_model_slow/batch_normalization_9/moving_variance',
        'resnet_model_slow/conv3d/kernel': 'resnet_model_slow/conv3d/kernel',
        'resnet_model_slow/conv3d_1/kernel': 'resnet_model_slow/conv3d_1/kernel',
        'resnet_model_slow/conv3d_10/kernel': 'resnet_model_slow/conv3d_10/kernel',
        'resnet_model_slow/conv3d_11/kernel': 'resnet_model_slow/conv3d_11/kernel',
        'resnet_model_slow/conv3d_12/kernel': 'resnet_model_slow/conv3d_12/kernel',
        'resnet_model_slow/conv3d_13/kernel': 'resnet_model_slow/conv3d_13/kernel',
        'resnet_model_slow/conv3d_14/kernel': 'resnet_model_slow/conv3d_14/kernel',
        'resnet_model_slow/conv3d_15/kernel': 'resnet_model_slow/conv3d_15/kernel',
        'resnet_model_slow/conv3d_16/kernel': 'resnet_model_slow/conv3d_16/kernel',
        'resnet_model_slow/conv3d_17/kernel': 'resnet_model_slow/conv3d_17/kernel',
        'resnet_model_slow/conv3d_18/kernel': 'resnet_model_slow/conv3d_18/kernel',
        'resnet_model_slow/conv3d_19/kernel': 'resnet_model_slow/conv3d_19/kernel',
        'resnet_model_slow/conv3d_2/kernel': 'resnet_model_slow/conv3d_2/kernel',
        'resnet_model_slow/conv3d_20/kernel': 'resnet_model_slow/conv3d_20/kernel',
        'resnet_model_slow/conv3d_21/kernel': 'resnet_model_slow/conv3d_21/kernel',
        'resnet_model_slow/conv3d_22/kernel': 'resnet_model_slow/conv3d_22/kernel',
        'resnet_model_slow/conv3d_23/kernel': 'resnet_model_slow/conv3d_23/kernel',
        'resnet_model_slow/conv3d_24/kernel': 'resnet_model_slow/conv3d_24/kernel',
        'resnet_model_slow/conv3d_25/kernel': 'resnet_model_slow/conv3d_25/kernel',
        'resnet_model_slow/conv3d_3/kernel': 'resnet_model_slow/conv3d_3/kernel',
        'resnet_model_slow/conv3d_4/kernel': 'resnet_model_slow/conv3d_4/kernel',
        'resnet_model_slow/conv3d_5/kernel': 'resnet_model_slow/conv3d_5/kernel',
        'resnet_model_slow/conv3d_6/kernel': 'resnet_model_slow/conv3d_6/kernel',
        'resnet_model_slow/conv3d_7/kernel': 'resnet_model_slow/conv3d_7/kernel',
        'resnet_model_slow/conv3d_8/kernel': 'resnet_model_slow/conv3d_8/kernel',
        'resnet_model_slow/conv3d_9/kernel': 'resnet_model_slow/conv3d_9/kernel',
        'resnet_model_slow/dense/bias': 'resnet_model_slow/dense/bias',
        'resnet_model_slow/dense/kernel': 'resnet_model_slow/dense/kernel'}


def vd_slowfast_a4_UCF():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_slowfast_a4"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_slowfast_a4_3"
    args['model_type'] = 'slowfast_a4'
    args["train_num_workers"] = 40
    #args["batch_size"] = 64
    #args["test_batch_size"] = 32
    args["lr_boundaries"] = '1370000,1390000'
    return args


def vd_slowfast_a4_UCF_pool1():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_slowfast_a4"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_slowfast_a4_pool1"
    args['final_pooling'] = 1
    args['model_type'] = 'slowfast_a4'
    args["train_num_workers"] = 40
    #args["batch_size"] = 64
    #args["test_batch_size"] = 32
    args["lr_boundaries"] = '1480000,1530000'
    return args


def vd_slowfast_pool1_scalecrop():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_slowfast_a4"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_slowfast_a4_pool1_sc"
    
    args['model_type'] = 'slowfast_a4'
    args['train_prep'] = 'MultiScaleCrop_224'
    args['test_no_frames'] = 5
    args['final_pooling'] = 1
    
    args['finetune_conv'] = True
    args["train_num_workers"] = 30

    args['init_lr'] = 0.0005
    args['weight_decay'] = 1e-5
    args['lr_boundaries'] = '1370000,1390000'
    return args

def vd_slowfast_a4_UCF_no_momentum():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_slowfast_a4"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_slowfast_a4_no_momentum_2"
    args['model_type'] = 'slowfast_a4'
    args["train_num_workers"] = 40
    #args["batch_size"] = 64
    #args["test_batch_size"] = 32
    args["lr_boundaries"] = '1370000,1390000'
    args['load_param_dict'] = slowfast_load_param_dict_no_momentum
    return args


def vd_slowfast_a4_all_UCF():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_slowfast_a4"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_slowfast_a4_all"
    args['model_type'] = 'slowfast_a4'
    args["train_num_workers"] = 100
    args["get_all_layers"] = '1,3,5,7,9'
    args["lr_boundaries"] = '1370000,1390000'
    return args


def vd_slowfast_single_all_UCF():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_slowfast_a4"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_slowfast_single_all"
    args['model_type'] = 'slowfastsingle_avg'
    args["get_all_layers"] = '5,7,9'
    args["train_num_workers"] = 100
    #args["from_ckpt"] = '/mnt/fs3/chengxuz/vd_relat/slowfast_single_model/model'
    args["test_batch_size"] = 32
    args["lr_boundaries"] = '1353000,1356000'
    return args


def vd_3dresnet_IR_UCF():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_3dresnet_IR"
    args['load_port'] = 27007
    #args['load_exp'] = None
    args["save_exp"] = "vd_finetune/UCF/vd_3dresnet"
    args['model_type'] = '3dresnet'
    args['train_prep'] = 'ColorJitter_112'
    args['finetune_conv'] = True
    args["train_num_workers"] = 40
    args['final_pooling'] = 1
    args['lr_boundaries'] = '1410000,1465000'
    return args


def vd_3dresnet_VIE_UCF():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_3dresnet"
    args['load_port'] = 27007
    #args['load_exp'] = None
    args["save_exp"] = "vd_finetune/UCF/vd_3dresnet_VIE"
    
    args['model_type'] = '3dresnet'
    args['train_prep'] = 'ColorJitter_112'
    args['test_no_frames'] = 10
    args['final_pooling'] = 1
    
    args['finetune_conv'] = True
    args["train_num_workers"] = 10
    args['lr_boundaries'] = '1555000,1615000'
    return args


def vd_3dresnet_VIE_sample():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_3dresnet"
    args['load_port'] = 27007
    #args['load_exp'] = None
    args["save_exp"] = "vd_finetune/UCF/vd_3dresnet_VIE_sample"
    
    args['model_type'] = '3dresnet'
    args['train_prep'] = 'MultiScaleCrop_112'
    args['test_no_frames'] = 10
    args['final_pooling'] = 1
    
    args['finetune_conv'] = True
    args["train_num_workers"] = 30

    args['init_lr'] = 0.05
    args["weight_decay"] = 5e-4
    args['lr_boundaries'] = '1535000,1575000'
    return args


def vd_3dresnet_VIE_sample_lr05_wd1():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_3dresnet"
    args['load_port'] = 27007
    #args['load_exp'] = None
    args["save_exp"] = "vd_finetune/UCF/vd_3dresnet_VIE_lr05_wd1"
    
    args['model_type'] = '3dresnet'
    args['train_prep'] = 'MultiScaleCrop_112'
    args['test_no_frames'] = 10
    args['final_pooling'] = 1
    
    args['finetune_conv'] = True
    args["train_num_workers"] = 30

    args['init_lr'] = 0.005
    args['lr_boundaries'] = '1505000,1540000'
    return args


def vd_3dresnet_VIE_sample_lr01_wd01():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_3dresnet"
    args['load_port'] = 27007
    #args['load_exp'] = None
    args["save_exp"] = "vd_finetune/UCF/vd_3dresnet_VIE_lr01_wd01"
    
    args['model_type'] = '3dresnet'
    args['train_prep'] = 'MultiScaleCrop_112'
    args['test_no_frames'] = 10
    args['final_pooling'] = 1
    
    args['finetune_conv'] = True
    args["train_num_workers"] = 30

    args['init_lr'] = 0.001
    args['weight_decay'] = 1e-5
    args['lr_boundaries'] = '1510000,1550000'
    return args


def vd_3dresnet_VIE_sample_lr01_wd01_drop():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_3dresnet"
    args['load_port'] = 27007
    #args['load_exp'] = None
    args["save_exp"] = "vd_finetune/UCF/vd_3dresnet_VIE_lr01_wd01_drop"
    
    args['model_type'] = '3dresnet'
    args['train_prep'] = 'MultiScaleCrop_112'
    args['test_no_frames'] = 10
    args['final_pooling'] = 1
    
    args['finetune_conv'] = True
    args["train_num_workers"] = 30

    args['init_lr'] = 0.001
    args['weight_decay'] = 1e-5
    args['lr_boundaries'] = '1476000,1482000'
    args['fre_valid'] = 500
    return args


def vd_3dresnet_VIE_sample_lr005_wd01():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_3dresnet"
    args['load_port'] = 27007
    #args['load_exp'] = None
    args["save_exp"] = "vd_finetune/UCF/vd_3dresnet_VIE_lr005_wd01"
    
    args['model_type'] = '3dresnet'
    args['train_prep'] = 'MultiScaleCrop_112'
    args['test_no_frames'] = 10
    args['final_pooling'] = 1
    
    args['finetune_conv'] = True
    args["train_num_workers"] = 30

    args['init_lr'] = 0.0005
    args['weight_decay'] = 1e-5
    args['lr_boundaries'] = '1480000,1485000'
    return args


def vd_3dresnet_VIE_sample_lr001_wd01():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_3dresnet"
    args['load_port'] = 27007
    #args['load_exp'] = None
    args["save_exp"] = "vd_finetune/UCF/vd_3dresnet_VIE_lr001_wd01"
    
    args['model_type'] = '3dresnet'
    args['train_prep'] = 'MultiScaleCrop_112'
    args['test_no_frames'] = 10
    args['final_pooling'] = 1
    
    args['finetune_conv'] = True
    args["train_num_workers"] = 30

    args['init_lr'] = 0.0001
    args['weight_decay'] = 1e-5
    args['lr_boundaries'] = '1500000,1530000'
    return args


def vd_3dresnet_VIE_sample_lr1_wd01():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_3dresnet"
    args['load_port'] = 27007
    #args['load_exp'] = None
    args["save_exp"] = "vd_finetune/UCF/vd_3dresnet_VIE_lr1_wd01"
    
    args['model_type'] = '3dresnet'
    args['train_prep'] = 'MultiScaleCrop_112'
    args['test_no_frames'] = 10
    args['final_pooling'] = 1
    
    args['finetune_conv'] = True
    args["train_num_workers"] = 30

    args['init_lr'] = 0.01
    args['weight_decay'] = 1e-5
    args['lr_boundaries'] = '1510000'
    return args


def vd_3dresnet_VIE_sample_lr01_wd1():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_3dresnet"
    args['load_port'] = 27007
    #args['load_exp'] = None
    args["save_exp"] = "vd_finetune/UCF/vd_3dresnet_VIE_lr01_wd1"
    
    args['model_type'] = '3dresnet'
    args['train_prep'] = 'MultiScaleCrop_112'
    args['test_no_frames'] = 10
    args['final_pooling'] = 1
    
    args['finetune_conv'] = True
    args["train_num_workers"] = 30

    args['init_lr'] = 0.001
    args['weight_decay'] = 1e-4
    args['lr_boundaries'] = '1510000,1540000'
    return args


def vd_3dresnet_IRUCF_UCF():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_3dresnet_IR_UCF"
    args['load_port'] = 27007
    #args['load_exp'] = None
    args["save_exp"] = "vd_finetune/UCF/vd_3dresnet_IRUCF"
    args['model_type'] = '3dresnet'
    args['train_prep'] = 'ColorJitter_112'
    args['finetune_conv'] = True
    args["train_num_workers"] = 10
    args['final_pooling'] = 1
    args['lr_boundaries'] = '525000'
    return args


def vd_3drotnet_UCF():
    args = {}

    args = finetune_basics(args)
    args = UCF_basics(args)
    args = finetune_UCF_bs128(args)

    args['load_exp'] = "vd_unsup_fx/rot/rot_3dresnet_re"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/UCF/vd_3drotnet_lr005"
    
    args['model_type'] = '3dresnet'
    args['train_prep'] = '3DRotNet_finetune'
    args['rotnet'] = True
    args['test_no_frames'] = 10
    args['final_pooling'] = 1
    
    args['finetune_conv'] = True
    args["train_num_workers"] = 30

    args['init_lr'] = 0.0005
    args['lr_boundaries'] = '185000'
    return args