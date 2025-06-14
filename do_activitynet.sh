#!/bin/bash
root_path=$1

exp_id="ac_DLDKD++"
collection=activitynet
visual_feature=i3d
model_name=DLDKD
device_ids=0
q_feat_size=1024
drop=0.25
input_drop=0.25
label_style=soft


CUDA_VISIBLE_DEVICES=0 python method/train.py --collection $collection --visual_feature $visual_feature \
                      --root_path $root_path --dset_name $collection \
                      --model_name $model_name --device_ids $device_ids  \
                      --distill_loss_decay exp --exp_id $exp_id\
                      --double_branch  --drop $drop --input_drop $input_drop  \
                      --q_feat_size $q_feat_size \
                      --label_style $label_style \
