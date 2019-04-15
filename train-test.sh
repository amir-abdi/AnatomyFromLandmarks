#!/bin/bash

filename=$(basename $0)

python3 src/main.py \
--postfix=${filename} \
--alg_name=gen3d \
--data_path="$MANDIBLE_DATA_PATH/train103" \
--labels_path="$MANDIBLE_DATA_PATH/train_landmarks103" \
--data_path_test="$MANDIBLE_DATA_PATH/test103" \
--labels_path_test="$MANDIBLE_DATA_PATH/test_landmarks103" \
--net_size=2.0 \
--batch_size=4 \
--bias=true \
--gen_g_loss=dice \
--num_fc=3 \
--split_ratio=0.80 \
--lr_gamma_g=0.99 \
--save_by_iter=false \
--image_save_step=300 \
--test=false \
--load_model_path=./results/pickle/model=model_alg=gen3d_postfix=train-test.sh/G_0.pkl \
--load_optim_path=./results/pickle/model=model_alg=gen3d_postfix=train-test.sh/G_optim_0.pkl \