# collection=$1
# visual_feature=$2
# root_path=$3
# model_dir=$4

# # training
# python method/eval.py  --collection $collection --visual_feature $visual_feature \
#                     --root_path $root_path  --dset_name $collection --model_dir $model_dir


python method/eval.py  --collection tvr --visual_feature i3d_resnet \
                    --root_path /home/zz/data  --dset_name tvr --model_dir tvr-double_dim384_kl_decay0.02_eval_3_7-2022_11_04_09_16_28
# python method/eval.py  --collection activitynet --visual_feature i3d \
#                     --root_path /home/zz/data  --dset_name activitynet --model_dir /home/zz/code/DL-DKD/results/activitynet-double_kl_8_ex_up_k800.0_loss_scale_weight1.0-2022_11_11_14_12_27