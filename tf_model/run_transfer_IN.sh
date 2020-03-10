# Directory hosting ImageNet tfrecords
image_dir=/mnt/fs1/Dataset/TFRecord_Imagenet_standard/image_label_full_widx
# Directory to host your saved models and logs
cache_dir=/mnt/fs4/chengxuz/video_pub_cache
# Your gpu number
gpu=0

main_setting=vd_transfer_IN.vd_3dresnet_trans_all
python train_transfer.py \
    --setting ${main_setting} \
    --cache_dir ${cache_dir} \
    --gpu ${gpu} \
    --image_dir ${image_dir} "$@"
