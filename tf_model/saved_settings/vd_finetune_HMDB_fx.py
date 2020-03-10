def finetune_HMDB_bs128(args):
    args['batch_size'] = 128
    args['test_batch_size'] = 64
    args['test_no_frames'] = 5
    args['fre_filter'] = 10000
    args['fre_cache_filter'] = 5000
    args['fre_valid'] = 500
    return args


def finetune_basics(args):
    args['port'] = 27007
    args['finetune_conv'] = True
    args['final_pooling'] = 1
    args['cache_dir'] = "/mnt/fs4/shetw/tfutils_cache"

    args['init_lr'] = 0.0005
    args['weight_decay'] = 1e-5
    return args

def HMDB_basics(args):
    args['image_dir'] = '/data5/shetw/HMDB51/extracted_frames'
    args['val_image_dir'] = args['image_dir']
    args['metafile_root'] = '/data5/shetw/HMDB51/metafiles'
    args['dataset'] = 'HMDB51'
    args['train_len'] = 3570
    args['val_len'] = 1530
    args['num_classes'] = 51
    return args

def finetune_HMDB_all_basics(args):
    args = finetune_basics(args)
    args = HMDB_basics(args)
    args = finetune_HMDB_bs128(args)
    return args


#################### Single-frame model ####################
def single_frame_setting(args):
    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_ctl"
    args['load_port'] = 27006
    args['test_no_frame'] = 5
    args["train_num_workers"] = 10
    return args

def vd_ctl_sc():
    args = {}
    args = finetune_HMDB_all_basics(args)
    args = single_frame_setting(args)

    args['train_prep'] = 'MultiScaleCrop_224'
    args["save_exp"] = "vd_finetune/HMDB/vd_ctl_sc_2"
    args["lr_boundaries"] = '1385000,1395000'
    return args

def vd_ctl_cj():
    args = {}
    args = finetune_HMDB_all_basics(args)
    args = single_frame_setting(args)

    args["save_exp"] = "vd_finetune/HMDB/vd_ctl_cj"
    args['train_prep'] = 'ColorJitter'
    args["lr_boundaries"] = '1385000,1395000'
    return args

def vd_sup_ctl_sc():
    args = {}
    args = finetune_HMDB_all_basics(args)
    args = single_frame_setting(args)    

    args['load_exp'] = "vd_sup/ctl/vd_f1_ctl"
    args["save_exp"] = "vd_finetune/HMDB/vd_sup_ctl_sc"
    args['train_prep'] = 'MultiScaleCrop_224'
    args["lr_boundaries"] = '715000,735000'
    return args

def vd_sup_ctl_cj():
    args = {}

    args = finetune_HMDB_all_basics(args)
    args = single_frame_setting(args)    
    args['load_exp'] = "vd_sup/ctl/vd_f1_ctl"
    args["save_exp"] = "vd_finetune/HMDB/vd_sup_ctl_cj"
    args['train_prep'] = 'ColorJitter'
    args["lr_boundaries"] = '710000,720000'
    return args


#################### Slowfast model ####################
def slowfast_setting(args):
    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_slowfast_a4"
    args['load_port'] = 27006
    args['model_type'] = 'slowfast_a4'
    args['test_no_frames'] = 2
    args["train_num_workers"] = 40
    return args

def vd_slowfast_a4_sc():
    args = {}

    args = finetune_HMDB_all_basics(args)
    args = slowfast_setting(args)
    args["save_exp"] = "vd_finetune/HMDB/vd_slowfast_sc"
    args['train_prep'] = 'MultiScaleCrop_224' 
    args["lr_boundaries"] = '1370000,1380000'
    return args

def vd_slowfast_a4_cj():
    args = {}

    args = finetune_HMDB_all_basics(args)
    args = slowfast_setting(args)
    args["save_exp"] = "vd_finetune/HMDB/vd_slowfast_cj"
    args['train_prep'] = 'ColorJitter'
    args["lr_boundaries"] = '1365000,1380000'
    return args

def vd_sup_slowfast_a4_sc():
    args = {}

    args = finetune_HMDB_all_basics(args)
    args = slowfast_setting(args)
    args['load_exp'] = "vd_sup/ctl/vd_sup_slowfast_a4"
    args["save_exp"] = "vd_finetune/HMDB/vd_sup_slowfast_sc"
    args['train_prep'] = 'MultiScaleCrop_224'  
    args["lr_boundaries"] = '340000'
    return args

def vd_sup_slowfast_a4_cj():
    args = {}

    args = finetune_HMDB_all_basics(args)
    args = slowfast_setting(args)
    args['load_exp'] = "vd_sup/ctl/vd_sup_slowfast_a4"
    args["save_exp"] = "vd_finetune/HMDB/vd_sup_slowfast_cj"
    args['train_prep'] = 'ColorJitter'
    args["lr_boundaries"] = '345000'
    return args


#################### Slowfast-single model ####################
def slowfast_single_setting(args):
    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_slowfast_a4"
    args["from_ckpt"] = '/mnt/fs3/chengxuz/vd_relat/slowfast_single_model/model'
    args['load_port'] = 27006
    args['model_type'] = 'slowfastsingle_avg'
    args["get_all_layers"] = '9'
    args['test_no_frames'] = 3
    args["train_num_workers"] = 40
    args['batch_size'] = 64
    args['slowfast_single_pooling'] = True
    return args

def vd_slowfast_single_a4_sc():
    args = {}

    args = finetune_HMDB_all_basics(args)
    args = slowfast_single_setting(args)
    args["save_exp"] = "vd_finetune/HMDB/vd_slowfast_single_sc_2"
    args['train_prep'] = 'MultiScaleCrop_224' 
    args["lr_boundaries"] = '10000,20000'
    return args

def vd_slowfast_single_a4_cj():
    args = {}

    args = finetune_HMDB_all_basics(args)
    args = slowfast_single_setting(args)
    args["save_exp"] = "vd_finetune/HMDB/vd_slowfast_single_cj_2"
    args['train_prep'] = 'ColorJitter'
    args["lr_boundaries"] = '10000,20000'
    return args


#################### 3D ResNet ####################
def resnet3d_setting(args):
    args['load_exp'] = "vd_unsup_fx/dyn_clstr/vd_3dresnet"
    args['load_port'] = 27007
    args['model_type'] = '3dresnet'
    args['test_no_frames'] = 4
    args["train_num_workers"] = 40  
    return args  

def vd_3dresnet_sc():
    args = {}
    args = finetune_HMDB_all_basics(args)
    args = resnet3d_setting(args)
    
    args["save_exp"] = "vd_finetune/HMDB/vd_3dresnet_sc"
    args['train_prep'] = 'MultiScaleCrop_112' 
    return args

def vd_3dresnet_cj():
    args = {}
    args = finetune_HMDB_all_basics(args)
    args = resnet3d_setting(args)
    
    args["save_exp"] = "vd_finetune/HMDB/vd_3dresnet_cj"
    args['train_prep'] = 'ColorJitter_112'  
    args['lr_boundaries'] = '1490000,1510000'
    return args

def vd_sup_3dresnet_sc():
    args = {}
    args = finetune_HMDB_all_basics(args)
    args = resnet3d_setting(args)

    args['load_exp'] = 'vd_sup/KN/vd_sup_3dresnet'
    args["save_exp"] = "vd_finetune/HMDB/vd_sup_3dresnet_sc"
    args['train_prep'] = 'MultiScaleCrop_112'  
    args['lr_boundaries'] = '140000'
    return args

def vd_sup_3dresnet_cj():
    args = {}
    args = finetune_HMDB_all_basics(args)
    args = resnet3d_setting(args)
    
    args['load_exp'] = 'vd_sup/KN/vd_sup_3dresnet'
    args["save_exp"] = "vd_finetune/HMDB/vd_sup_3dresnet_cj"
    args['train_prep'] = 'ColorJitter_112'  
    args['lr_boundaries'] = '140000'
    return args

def vd_opn_sc():
    args = {}
    args = finetune_HMDB_all_basics(args)
    
    args['load_exp'] = "vd_unsup_fx/opn/opn_random_sep_flow"
    args['load_port'] = 27006
    args['test_no_frames'] = 5
    args["train_num_workers"] = 20
    
    args["save_exp"] = "vd_finetune/HMDB/vd_opn_sc"
    args['train_prep'] = 'MultiScaleCrop_224'  
    #args['lr_boundaries'] = '290000+'
    args['lr_boundaries'] = '340000'
    return args




def vd_3drotnet_HMDB():
    args = {}

    args = finetune_HMDB_all_basics(args)

    args['load_exp'] = "vd_unsup_fx/rot/rot_3dresnet"
    args['load_port'] = 27006
    args["save_exp"] = "vd_finetune/HMDB/vd_3drotnet_hflip"
    
    args['model_type'] = '3dresnet'
    args['train_prep'] = '3DRotNet_finetune'
    args['test_no_frames'] = 10
    args['final_pooling'] = 1
    
    args['finetune_conv'] = True
    args["train_num_workers"] = 30

    args['init_lr'] = 0.008
    #args['lr_boundaries'] = '246000'
    return args
