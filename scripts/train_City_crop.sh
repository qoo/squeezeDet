#!/bin/bash

export GPUID=0
export NET="squeezeDet"
export TRAIN_DIR="/home/cyeh/logs/squeezeDet/test/"


case "$NET" in
  "squeezeDet")
    export PRETRAINED_MODEL_PATH="./data/SqueezeNet/squeezenet_v1.1.pkl"
    ;;
  "squeezeDet+")
    export PRETRAINED_MODEL_PATH="./data/SqueezeNet/squeezenet_v1.0_SR_0.750.pkl"
    ;;
  "resnet50")
    export PRETRAINED_MODEL_PATH="./data/ResNet/ResNet-50-weights.pkl"
    ;;
  "vgg16")
    export PRETRAINED_MODEL_PATH="./data/VGG16/VGG_ILSVRC_16_layers_weights.pkl"
    ;;
  *)
    echo "net architecture not supported."
    exit 0
    ;;
esac


python ./src/train.py \
  --dataset=KITTI \
  --pretrained_model_path=$PRETRAINED_MODEL_PATH \
  --data_path=./data/KITTI_Cityscape_crop \
  --image_set=train \
  --train_dir="$TRAIN_DIR/train" \
  --net=$NET \
  --summary_step=100 \
  --checkpoint_step=500 \
  --gpu=$GPUID

