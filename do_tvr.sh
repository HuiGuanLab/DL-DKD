#!/bin/bash

root_path=$1

exp_id="tvr_DLDKD++"
collection=tvr
visual_feature=i3d_resnet
q_feat_size=768
model_name=DLDKD
margin=0.1
device_ids=0
n_heads=4
lr=0.0003
drop=0.2
input_drop=0.2
label_style=soft


CUDA_VISIBLE_DEVICES=2 python method/train.py --collection $collection --visual_feature $visual_feature \
                  --root_path $root_path --dset_name $collection \
                  --q_feat_size $q_feat_size --model_name $model_name \
                  --margin $margin --device_ids $device_ids \
                  --n_heads $n_heads --distill_loss_decay exp \
                  --double_branch --drop $drop --input_drop $input_drop --lr $lr  \
                  --label_style $label_style 