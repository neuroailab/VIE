# Directory hosting your extracted frames
image_dir=/data5/shetw/UCF101/extracted_frames
# Directory to host your saved models and logs
cache_dir=/mnt/fs4/chengxuz/video_pub_cache
# If you are using metas provided by us, this should be /repo_dir/build_data/kinetics
meta_dir=/home/chengxuz/video_unsup/build_data/UCF101
# Your gpu number
gpu=0

main_setting=vd_finetune_UCF_fx.vd_3dresnet_sc # Table 2 result
#main_setting=vd_finetune_UCF_fx.vd_3dresnet_cj # Table 3 result

python train_transfer_KN.py \
    --setting ${main_setting} \
    --cache_dir ${cache_dir} \
    --gpu ${gpu} \
    --metafile_root ${meta_dir} \
    --image_dir ${image_dir} \
    --val_image_dir ${image_dir} "$@"
