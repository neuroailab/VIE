# Instructions for training

We first show how to train a VIE-3DResNet as an example, including pretraining on Kinetics, transfer learning on ImageNet and Kinetics, and fine-tuning on UCF101 and HMDB51. 
We then show how other models can be trained with minor modifications to the commands of training VIE-3DResNet.

Due to legacy reasons, we only support tensorflow < 2.0. We have tested our codes with Python 3.7. 
You also need to install [faiss-gpu](https://github.com/facebookresearch/faiss)==1.6.1 and pytorch.

## VIE-3DResNet
You can either start the training directly only using one gpu or do the training with the help of tfutils, which provides support for multi-gpu training.

### Training directly

Make required changes to the following script to set the parameters and then run it to start the training including first starting an VIE-3DResNet-IR training as pretraining for the VIE-3DResNet and then starting the main VIE-3DResNet training:
```
sh run_training.sh
```

### Transfer learning to Kinetics and ImageNet; fine-tuning to UCF101 and HMDB51
For transfer learning to ImageNet, you need to first build ImageNet tfrecords following instructions in [LocalAggregation](https://github.com/neuroailab/LocalAggregation.git) repo. Then, run the following script:
```
sh run_transfer_IN.sh
```

For transfer learning to Kinetics:
```
sh run_transfer_KN.sh
```

For finetuning experiments, we provide configs for reproducing both Table 2 and Table 3 results in the scripts, by default, the script will reproduce the Table 2 result. See the following scripts for how to generate the Table 3 result.
For finetuning to UCF101:
```
sh run_finetune_UCF.sh
```

For finetuning to HMDB51:
```
sh run_finetune_HMDB.sh
```

### Training using tfutils
Install `tfutils` as following.
```
git clone https://github.com/neuroailab/tfutils.git
cd tfutils
python setup.py install --user
```

For all the previuos training examples, just add ` --tfutils --port mongodb_port_number` after the command to start the training using tfutils, such as `sh run_training.sh --tfutils --port mongodb_port_number`.
However, you need to start a mongodb for tfutils to work.


## Other models
COMING SOON
