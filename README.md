# Unsupervised Learning from Video with Deep Neural Embeddings

Please see codes in `build_data` to prepare different datasets, you need to have kinetics at least to run the training. 
After that, please see codes in `tf_model` to train the model and evaluate it.
Finally, check `show_results.ipynb` in `notebook` folder to see how the training progress can be checked and compared to our training trajectory.

## Pretrained weights for VIE-3DResNet (updated 12/31/2020)

Weights can be downloaded at [this link](http://visualmaster-models.s3.amazonaws.com/vie/3dresnet_112/checkpoint-1450000.tar).

## How to get responses from intermediate layers

Check function `test_video_model` in script `tf_model/generate_resps_from_ckpt.py`.
The outputs will be stored in a dictionary, with keys like `encode_x` (x is from 1 to 10).
Layer `encode_1` is the output of the first pooling layer.
The other layers are outputs from the following residual blocks (ResNet18 has 9 residual blocks in total).
The output is in shape `(batch_size, channels, temporal_dim, spatial_dim, spatial_dim)`.
