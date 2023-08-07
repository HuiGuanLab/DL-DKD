collection=tvr
visual_feature=i3d_resnet
q_feat_size=768
model_name=MS_SL_Net
margin=0.1
root_path=/home/zms/VisualSearch
device_ids=0
frame_weight=0.5
clip_weight=0.5
A_hidden_size=384
B_hidden_size=384
hard_negative_start_epoch=0
n_heads=4
use_clip=true
double_branch=true
decay_way=1
loss_init_weight=0.1 #用于权重上升的时候增加的值
loss_scale_weight=0.1
linear_b=1
linear_k=-0.01
sigmoid_k=800
exponential_k=0.95
# training
for decay_way in 1
do
  CUDA_VISIBLE_DEVICES=1 python method/train.py  --collection $collection --visual_feature $visual_feature \
                      --root_path $root_path  --dset_name $collection \
                      --q_feat_size $q_feat_size --model_name $model_name \
                      --margin $margin --device_ids $device_ids \
                      --B_hidden_size $B_hidden_size --n_heads $n_heads --A_hidden_size $A_hidden_size \
                      --frame_weight $frame_weight --clip_weight $clip_weight \
                      --use_clip $use_clip --double_branch $double_branch --hard_negative_start_epoch $hard_negative_start_epoch \
                      --loss_init_weight $loss_init_weight --decay_way $decay_way \
                      --linear_k $linear_k --sigmoid_k $sigmoid_k --linear_b $linear_b \
                      --exponential_k $exponential_k --loss_scale_weight $loss_scale_weight

done