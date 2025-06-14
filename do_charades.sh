#!/bin/bash

root_path=$1

exp_id="charades_DLDKD++"
collection=charades
visual_feature=i3d_rgb_lgi
model_name=DLDKD
device_ids=0
q_feat_size=1024
drop=0.15
input_drop=0.15
lr=0.00024
label_style=soft


CUDA_VISIBLE_DEVICES=1 python method/train.py --collection $collection --visual_feature $visual_feature \
                      --root_path $root_path --dset_name $collection \
                      --model_name $model_name --device_ids $device_ids --lr $lr \
                      --distill_loss_decay exp   --exp_id $exp_id \
                      --double_branch  --q_feat_size $q_feat_size  \
                      --drop $drop --input_drop $input_drop \
                       --label_style $label_style \
                      