def sup_basic(args):
    args['port'] = 27007
    args['db_name'] = 'vd_sup'
    args['cache_dir'] = '/mnt/fs4/shetw/tfutils_cache'
    args['task'] = 'SUP'
    return args

def sup_bs128(args):
    args['batch_size'] = 128
    args['test_batch_size'] = 64
    args['test_no_frames'] = 5
    args['fre_filter'] = 10000
    args['fre_cache_filter'] = 5000
    args['fre_valid'] = 500
    
    args['init_lr'] = 0.01
    return args

def UCF_basics(args):
    args = sup_basic(args)
    args = sup_bs128(args)

    args['col_name'] = 'UCF'
    args['data_len'] = 9537
    args['image_dir'] = '/data5/shetw/UCF101/extracted_frames'
    args['val_image_dir'] = args['image_dir']
    args['metafile_root'] = '/data5/shetw/UCF101/metafiles'
    args['dataset'] = 'UCF101'
    args['train_len'] = 9537
    args['val_len'] = 3783
    args['num_classes'] = 101
    return args


def HMDB_basics(args):
    args = sup_basic(args)
    args = sup_bs128(args)

    args['col_name'] = 'HMDB'
    args['image_dir'] = '/data5/shetw/HMDB51/extracted_frames'
    args['val_image_dir'] = args['image_dir']
    args['metafile_root'] = '/data5/shetw/HMDB51/metafiles'
    args['dataset'] = 'HMDB51'
    args['train_len'] = 3570
    args['data_len'] = 3570
    args['val_len'] = 1530
    args['train_prep'] = 'ColorJitter'
    args['num_classes'] = 51
    return args

#################### Single-frame model ####################
def single_frame_setting(args):
    args['test_no_frame'] = 5
    args["train_num_workers"] = 10
    return args

def vd_single_UCF_sc():
    args = {}
    args = UCF_basics(args)
    args = single_frame_setting(args)

    args['train_prep'] = 'MultiScaleCrop_224'
    args['exp_id'] = 'vd_single_sc'
    args['lr_boundaries'] = '9000,13000'
    return args

def vd_single_UCF_cj():
    args = {}
    args = UCF_basics(args)
    args = single_frame_setting(args)

    args['train_prep'] = 'ColorJitter'
    args['exp_id'] = 'vd_single_cj'
    args['lr_boundaries'] = '9000,13000'
    return args

def vd_single_HMDB_sc():
    args = {}
    args = HMDB_basics(args)
    args = single_frame_setting(args)

    args['train_prep'] = 'MultiScaleCrop_224'
    args['exp_id'] = 'vd_single_sc'
    args['lr_boundaries'] = '7000,10000'
    return args

def vd_single_HMDB_cj():
    args = {}
    args = HMDB_basics(args)
    args = single_frame_setting(args)

    args['train_prep'] = 'ColorJitter'
    args['exp_id'] = 'vd_single_cj'
    args['lr_boundaries'] = '7000,10000'
    return args


#################### Slowfast model ####################
def slowfast_setting(args):
    args['model_type'] = 'slowfast_a4'
    args["train_num_workers"] = 40
    return args

def vd_slowfast_UCF_sc():
    args = {}
    args = UCF_basics(args)
    args = slowfast_setting(args)

    args['test_no_frames'] = 3
    args['train_prep'] = 'MultiScaleCrop_224'
    args['exp_id'] = 'vd_slowfast_sc'
    #args['lr_boundaries'] = '390011,590011'
    return args

def vd_slowfast_UCF_cj():
    args = {}
    args = UCF_basics(args)
    args = slowfast_setting(args)

    args['test_no_frames'] = 3
    args['train_prep'] = 'ColorJitter'
    args['exp_id'] = 'vd_slowfast_cj'
    args['lr_boundaries'] = '10000,14000'
    return args

def vd_slowfast_HMDB_sc():
    args = {}
    args = HMDB_basics(args)
    args = slowfast_setting(args)

    args['test_no_frames'] = 2
    args['train_prep'] = 'MultiScaleCrop_224'
    args['exp_id'] = 'vd_slowfast_sc'
    args['lr_boundaries'] = '13000,18000'
    return args

def vd_slowfast_HMDB_cj():
    args = {}
    args = HMDB_basics(args)
    args = slowfast_setting(args)

    args['test_no_frames'] = 2
    args['train_prep'] = 'ColorJitter'
    args['exp_id'] = 'vd_slowfast_cj'
    args['lr_boundaries'] = '9000,13000'
    return args


#################### 3D ResNet ####################
def resnet3d_setting(args):
    args['model_type'] = '3dresnet'
    args["train_num_workers"] = 40  
    return args  

def vd_resnet3d_UCF_sc():
    args = {}
    args = UCF_basics(args)
    args = resnet3d_setting(args)

    args['test_no_frames'] = 10
    args['train_prep'] = 'MultiScaleCrop_112'
    args['exp_id'] = 'vd_resnet3d_sc'
    args['lr_boundaries'] = '10000,16000'
    return args

def vd_resnet3d_UCF_cj():
    args = {}
    args = UCF_basics(args)
    args = resnet3d_setting(args)

    args['test_no_frames'] = 10
    args['train_prep'] = 'ColorJitter_112'
    args['exp_id'] = 'vd_resnet3d_cj'
    #args['lr_boundaries'] = '390011,590011'
    return args

def vd_resnet3d_HMDB_sc():
    args = {}
    args = HMDB_basics(args)
    args = resnet3d_setting(args)

    args['test_no_frames'] = 4
    args['train_prep'] = 'MultiScaleCrop_112'
    args['exp_id'] = 'vd_resnet3d_sc'
    args['lr_boundaries'] = '10000,15000'
    return args

def vd_resnet3d_HMDB_cj():
    args = {}
    args = HMDB_basics(args)
    args = resnet3d_setting(args)

    args['test_no_frames'] = 4
    args['train_prep'] = 'ColorJitter_112'
    args['exp_id'] = 'vd_resnet3d_cj'
    args['lr_boundaries'] = '10000,15000'
    return args