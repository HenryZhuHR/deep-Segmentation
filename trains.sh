deeplab_model_list=("deeplabv3plus_xception" "deeplabv3plus_resnet50")  #"deeplabv3plus_hrnetv2_32")
chekpoints_dir="checkpoints"
for deeplab_model in ${deeplab_model_list[@]};
do
    echo "${chekpoints_dir}/$deeplab_model"
    python train.py \
        --num_classes 3 \
        --model $deeplab_model \
        --device "cuda:0" \
        --train_batch_size 4 \
        --valid_batch_size 4 \
        --num_workers 0 \
        --epochs 50 \
        --save_dir "${chekpoints_dir}/$deeplab_model" \
        --save_name $deeplab_model;
done