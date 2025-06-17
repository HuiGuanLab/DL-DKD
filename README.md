# DL-DKD++
<!-- Source code of our ICCV 2023 paper [Dual Learning with Dynamic Knowledge Distillation for  Partially Relevant Video Retrieval](). -->

<!-- Homepage of our paper [http://danieljf24.github.io/prvr/](http://danieljf24.github.io/prvr/). -->

<img src="figures/DLDKD++.png" width="1100px">

## Table of Contents

* [Environments](#environments)
* [DL-DKD++ on TVR](#DL-DKD++-on-TVR)
  * [Required Data](#Required-Data)
  * [Model Training](#Training)
  * [Model Evaluation](#Evaluation)
  * [Expected Performance](#Expected-Performance)
* [DL-DKD++ on Activitynet](#DL-DKD++-on-activitynet)
  * [Required Data](#Required-Data-1)
  * [Model Training](#Training-1)
  * [Model Evaluation](#Evaluation-1)
  * [Expected Performance](#Expected-Performance-1)
* [DL-DKD++ on Charades-STA](#DL-DKD++-on-Charades-STA)
  * [Required Data](#Required-Data-2)
  * [Model Training](#Training-2)
  * [Model Evaluation](#Evaluation-2)
  * [Expected Performance](#Expected-Performance-2)
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
git clone https://github.com/HuiGuanLab/DL-DKD.git 待修改
cd DL-DKD
pip install -r requirements.txt
conda deactivate
```

## DL-DKD++ on TVR

### Required Data
Run the following script to download the video feature and text feature of the TVR dataset and place them in the specified path. The data can also be downloaded from [Kuake pan](https://pan.quark.cn/s/8086102776f4). Please refer to [here](https://github.com/HuiGuanLab/ms-sl/tree/main/dataset) for more description of the dataset.



### Training
Run the following script to train `DL-DKD++` network on TVR. It will save the chechpoint that performs best on the validation set as the final model.

```
root_path=$HOME/VisualSearch
mkdir -p $root_path && cd $root_path
unzip tvr.zip
```

```
#Add project root to PYTHONPATH (Note that you need to do this each time you start a new session.)
source setup.sh

conda activate DLDKD

root_path=$HOME/VisualSearch

./do_tvr.sh $root_path
```

### Evaluation
The model is placed in the directory $root\_path/results/$collection/$model_dir after training. To evaluate it, please run the following script:
```
collection=tvr
visual_feature=i3d_resnet
root_path=$HOME/VisualSearch
model_dir=tvr_DLDKD++

./do_test.sh $collection $visual_feature $root_path $model_dir
```

We also provide the trained checkpoint on TVR, run the following script to evaluate it. The model can also be downloaded from [Here](https://pan.quark.cn/s/cad7328a194b)， passward=daNQ. 
```
collection=tvr
visual_feature=i3d_resnet
root_path=$HOME/VisualSearch
model_dir=tvr/tvr_DLDKD++

unzip DLDKD++_checkpoint.zip -d $ROOTPATH/

./do_test.sh $collection $visual_feature $root_path $model_dir
```
`$collection` is the dataset that the model trained and evaluate on.

`$visual_feature` is the video feature corresponding to the dataset.

`$model_dir` is the path of checkpoints saved.
### Expected performance 

|             | R@1  | R@5  | R@10 | R@100 | SumR  |
| :---------: | :--: | :--: | :--: | :---: | :---: |
| Text-to-Video | 15.3 | 36.0 | 47.5 | 86.0  | 184.8 |

## DL-DKD++ on Activitynet
### Required Data
Run the following script to download the video feature and text feature of the Activitynet dataset and place them in the specified path. The data can also be downloaded from [Kuake pan](https://pan.quark.cn/s/4df1583afa13). Please refer to [here](https://github.com/HuiGuanLab/ms-sl/tree/main/dataset) for more description of the dataset.

```
root_path=$HOME/VisualSearch
mkdir -p $root_path && cd $root_path
unzip activitynet.zip
```

### Training
Run the following script to train `DL-DKD++` network on Activitynet.
```
#Add project root to PYTHONPATH (Note that you need to do this each time you start a new session.)
source setup.sh

conda activate DLDKD

root_path=$HOME/VisualSearch

./do_activitynet.sh $root_path
```

### Evaluation
The model is placed in the directory $root\_path/$collection/results/$model_dir after training. To evaluate it, please run the following script:
```
collection=activitynet
visual_feature=i3d
root_path=$HOME/VisualSearch
model_dir=ac_DLDKD++

./do_test.sh $collection $visual_feature $root_path $model_dir
```

We also provide the trained checkpoint on Activitynet, run the following script to evaluate it. The model can also be downloaded from [Here](https://pan.quark.cn/s/cad7328a194b)， passward=daNQ.
```
collection=activitynet
visual_feature=i3d
root_path=$HOME/VisualSearch
model_dir=activitynet/ac_DLDKD++

unzip DLDKD++_checkpoint.zip -d $ROOTPATH/

./do_test.sh $collection $visual_feature $root_path $model_dir
```

### Expected performance 

|             | R@1  | R@5  | R@10 | R@100 | SumR  |
| :---------: | :--: | :--: | :--: | :---: | :---: |
| Text-to-Video | 8.3 | 25.5 | 38.3 | 77.8  | 149.9 |

## DL-DKD++ on Charades-STA
### Required Data
Run the following script to download the video feature and text feature of the Charades-STA dataset and place them in the specified path. The data can also be downloaded from [Baidu pan](https://pan.baidu.com/s/1UNu67hXCbA6ZRnFVPVyJOA?pwd=8bh4). Please refer to [here](https://github.com/HuiGuanLab/ms-sl/tree/main/dataset) for more description of the dataset.

```
root_path=$HOME/VisualSearch
mkdir -p $root_path && cd $root_path
unzip charades.zip -d $root_path
```

### Training
Run the following script to train `DL-DKD++` network on Charades-STA.
```
#Add project root to PYTHONPATH (Note that you need to do this each time you start a new session.)
source setup.sh

conda activate DLDKD

root_path=$HOME/VisualSearch

./do_charades.sh $root_path
```

### Evaluation
The model is placed in the directory $root\_path/$collection/results/$model_dir after training. To evaluate it, please run the following script:
```
collection=charades
visual_feature=i3d
root_path=$HOME/VisualSearch
model_dir=charades_DLDKD++

./do_test.sh $collection $visual_feature $root_path $model_dir
```

We also provide the trained checkpoint on Charades-STA, run the following script to evaluate it. The model can also be downloaded from [Here](https://pan.quark.cn/s/cad7328a194b)， passward=daNQ.
```
collection=charades
visual_feature=i3d
root_path=$HOME/VisualSearch
model_dir=charades/charades_DLDKD++

unzip DLDKD++_checkpoint.zip -d $ROOTPATH/

./do_test.sh $collection $visual_feature $root_path $model_dir
```

### Expected performance 

|             | R@1  | R@5  | R@10 | R@100 | SumR  |
| :---------: | :--: | :--: | :--: | :---: | :---: |
| Text-to-Video | 1.9 | 7.1 | 12.3 | 49.8  | 71.1 |

以下待修改
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
The codes are modified from [TVRetrieval](https://github.com/jayleicn/TVRetrieval) and [ReLoCLNet](https://github.com/IsaacChanghau/ReLoCLNet).

This work was supported by the National Key R&D Program of China (2018YFB1404102), NSFC (62172420,61902347, 61976188, 62002323), the Public Welfare Technology Research Project of Zhejiang Province (LGF21F020010), the Open Projects Program of the National Laboratory of Pattern Recognition, the Fundamental Research Funds for the Provincial Universities of Zhejiang, and Public Computing Cloud of RUC.
