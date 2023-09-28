# DL-DKD
Source code of our ICCV 2023 paper [Dual Learning with Dynamic Knowledge Distillation for  Partially Relevant Video Retrieval]().

<!-- Homepage of our paper [http://danieljf24.github.io/prvr/](http://danieljf24.github.io/prvr/). -->

<img src="https://github.com/HuiGuanLab/DL-DKD/blob/master/figures/DLDKD_model.png" width="1100px">

## Table of Contents

* [Environments](#environments)
* [DL-DKD on TVR](#DL-DKD-on-TVR)
  * [Required Data](#Required-Data)
  * [Model Training](#Training)
  * [Model Evaluation](#Evaluation)
  * [Expected Performance](#Expected-Performance)
* [DL-DKD on Activitynet](#DL-DKD-on-activitynet)
  * [Required Data](#Required-Data-1)
  * [Model Training](#Training-1)
  * [Model Evaluation](#Evaluation-1)
  * [Expected Performance](#Expected-Performance-1)
* [Reference](#Reference)
* [Acknowledgement](#Acknowledgement)

## Environments 
* **python 3.8**
* **pytorch 1.9.0**
* **torchvision 0.10.0**
* **tensorboard 2.6.0**
* **tqdm 4.62.0**
* **easydict 1.9**
* **h5py 2.10.0**
* **cuda 11.1**

We used Anaconda to setup a deep learning workspace that supports PyTorch. Run the following script to install the required packages.
```
conda create --name DLDKD python=3.8
conda activate DLDKD
git clone https://github.com/HuiGuanLab/DL-DKD.git
cd DL-DKD
pip install -r requirements.txt
conda deactivate
```

## DL-DKD on TVR

### Required Data
Run the following script to download the video feature and text feature of the TVR dataset and place them in the specified path. The data can also be downloaded from [Kuake pan](https://pan.quark.cn/s/8fad55178323). Please refer to [here](https://github.com/HuiGuanLab/ms-sl/tree/main/dataset) for more description of the dataset.



### Training
Run the following script to train `DL-DKD` network on TVR. It will save the chechpoint that performs best on the validation set as the final model.

```
ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH && cd $ROOTPATH
unzip activitynet.zip
```

```
#Add project root to PYTHONPATH (Note that you need to do this each time you start a new session.)
source setup.sh

conda activate DLDKD

ROOTPATH=$HOME/VisualSearch

./do_tvr.sh $ROOTPATH
```

### Evaluation
The model is placed in the directory $ROOTPATH/$DATASET/results/$MODELDIR after training. To evaluate it, please run the following script:
```
DATASET=tvr
FEATURE=i3d_resnet
ROOTPATH=$HOME/VisualSearch
MODELDIR=tvr-double_dim384_kl_decay0.02_eval_3_7-2022_11_04_09_16_28

./do_test.sh $DATASET $FEATURE $ROOTPATH $MODELDIR
```

We also provide the trained checkpoint on TVR, run the following script to evaluate it. The model can also be downloaded from [Here](https://pan.quark.cn/s/3fc1aba283d5). 
```
DATASET=tvr
FEATURE=i3d_resnet
ROOTPATH=$HOME/VisualSearch
MODELDIR=checkpoint_tvr

unzip checkpoint_tvr.zip -d $ROOTPATH/$DATASET/results

./do_test.sh $DATASET $FEATURE $ROOTPATH $MODELDIR
```
`$DATASET` is the dataset that the model trained and evaluate on.

`$FEATURE` is the video feature corresponding to the dataset.

`$MODELDIR` is the path of checkpoints saved.
### Expected performance 

|             | R@1  | R@5  | R@10 | R@100 | SumR  |
| :---------: | :--: | :--: | :--: | :---: | :---: |
| Text-to-Video | 14.4 | 34.9 | 45.8 | 84.9  | 179.9 |

## DL-DKD on Activitynet
### Required Data
Run the following script to download the video feature and text feature of the Activitynet dataset and place them in the specified path. The data can also be downloaded from [Kuake pan](https://pan.quark.cn/s/0fc241b533d6). Please refer to [here](https://github.com/HuiGuanLab/ms-sl/tree/main/dataset) for more description of the dataset.

```
ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH && cd $ROOTPATH
unzip activitynet.zip
```

### Training
Run the following script to train `DL-DKD` network on Activitynet.
```
#Add project root to PYTHONPATH (Note that you need to do this each time you start a new session.)
source setup.sh

conda activate DLDKD

ROOTPATH=$HOME/VisualSearch

./do_activitynet.sh $ROOTPATH
```

### Evaluation
The model is placed in the directory $ROOTPATH/$DATASET/results/$MODELDIR after training. To evaluate it, please run the following script:
```
DATASET=activitynet
FEATURE=i3d
ROOTPATH=$HOME/VisualSearch
MODELDIR=activitynet-double_kl_8_ex_up_k800.0_loss_scale_weight1.0-2022_11_11_14_12_27

./do_test.sh $DATASET $FEATURE $ROOTPATH $MODELDIR
```

We also provide the trained checkpoint on Activitynet, run the following script to evaluate it. The model can also be downloaded from [Here](https://pan.quark.cn/s/d55c7937e74e).
```
DATASET=activitynet
FEATURE=i3d
ROOTPATH=$HOME/VisualSearch
MODELDIR=checkpoint_activitynet

unzip checkpoint_activitynet.zip -d $ROOTPATH/$DATASET/results

./do_test.sh $DATASET $FEATURE $ROOTPATH $MODELDIR
```

### Expected performance 

|             | R@1  | R@5  | R@10 | R@100 | SumR  |
| :---------: | :--: | :--: | :--: | :---: | :---: |
| Text-to-Video | 8.0 | 25.0 | 37.5 | 77.1  | 147.6 |



## Reference
```
@inproceedings{dong2023DLDKD,
title = {Dual Learning with Dynamic Knowledge Distillation for Partially Relevant Video Retrieval},
author = {Jianfeng Dong and Minsong Zhang and Zheng Zhang and Xianke Chen and Daizong Liu and Xiaoye Qu and Xun Wang and Baolong Liu},
booktitle = {IEEE International Conference on Computer Vision},
year = {2023},
}
```
## Acknowledgement
The codes are modified from [TVRetrieval](https://github.com/jayleicn/TVRetrieval),[ReLoCLNet](https://github.com/IsaacChanghau/ReLoCLNet) and [MS-SL](\(https://github.com/HuiGuanLab/ms-sl).

This work was supported by the ``Pioneer" and ``Leading Goose" R\&D Program of Zhejiang (No.2023C01212), National Natural Science Foundation of China (No. 61976188), Young Elite Scientists Sponsorship Program by CAST (No. 2022QNRC001), the open research fund of The State Key Laboratory of Multimodal Artificial Intelligence Systems, and the Fundamental Research Funds for the Provincial Universities of Zhejiang.
