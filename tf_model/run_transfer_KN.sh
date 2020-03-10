# Directory hosting your extracted frames
image_dir=/data5/chengxuz/Dataset/kinetics/comp_jpgs_extracted
# Directory to host your saved models and logs
cache_dir=/mnt/fs4/chengxuz/video_pub_cache
# If you are using metas provided by us, this should be /repo_dir/build_data/kinetics
meta_dir=/home/chengxuz/video_unsup/build_data/kinetics
# Your gpu number
gpu=0

main_setting=vd_transfer_KN.vd_3dresnet_trans_all
python train_transfer_KN.py \
    --setting ${main_setting} \
    --cache_dir ${cache_dir} \
    --gpu ${gpu} \
    --metafile_root ${meta_dir} \
    --image_dir ${image_dir} \
    --val_image_dir ${image_dir} "$@"
