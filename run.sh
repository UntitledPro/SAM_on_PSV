#!/usr/bin/env bash


# export CUDA_VISIBLE_DEVICES=3
# nohup python -u finetune.py \
#     --ds_size 0.05 \
#     --num_epochs 1000 \
#     --interval 100 \
#     --ckp_dir 'work_dirs/sam_psv/' \
#     --ckp_name 'pair_med_loss_05' \
#     --critn 'med' \
#     --pair \
#     >logs/pair_med_loss_05.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=0
# nohup python -u finetune.py \
#     --ds_size 0.05 \
#     --num_epochs 1000 \
#     --interval 100 \
#     --ckp_dir 'work_dirs/sam_psv/' \
#     --ckp_name 'med_loss_05_10' \
#     --resume \
#     --critn 'med' \
#     --noise_rate 1 \
#     >logs/med_loss_05_10.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=1
# nohup python -u finetune.py \
#     --ds_size 0.1 \
#     --num_epochs 500 \
#     --interval 50 \
#     --ckp_dir 'work_dirs/sam_psv/' \
#     --ckp_name 'med_loss_10_10' \
#     --resume \
#     --critn 'med' \
#     --noise_rate 1 \
#     >logs/med_loss_10_10.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=2
# nohup python -u finetune.py \
#     --ds_size 0.2 \
#     --num_epochs 250 \
#     --interval 25 \
#     --ckp_dir 'work_dirs/sam_psv/' \
#     --ckp_name 'med_loss_20_10' \
#     --resume \
#     --critn 'med' \
#     --noise_rate 1 \
#     >logs/med_loss_20_10.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=0
# nohup python -u finetune.py \
#     --ckp_dir 'work_dirs/sam_psv/' \
#     --ckp_name 'sam_loss_full' \
#     --resume \
#     --critn 'sam' \
#     >logs/sam_loss_full.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=0
# nohup python -u finetune.py \
#     --ds_size 0.5 \
#     --num_epochs 100 \
#     --interval 10 \
#     --ckp_dir 'work_dirs/sam_psv/' \
#     --ckp_name 'med_loss_50_40' \
#     --resume \
#     --critn 'med' \
#     --noise_rate 6 \
#     >logs/med_loss_50_40.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=1
# nohup python -u finetune.py \
#     --mode 'full_finetune' \
#     --num_epochs 50 \
#     --interval 5 \
#     --ckp_dir 'work_dirs/sam_psv/' \
#     --ckp_name 'med_loss_full_ana' \
#     --resume \
#     --critn 'med' \
#     >logs/med_loss_full_ana.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=0
# nohup python -u finetune.py \
#     --mode 'finetune' \
#     --num_epochs 50 \
#     --interval 5 \
#     --ckp_dir 'work_dirs/sam_psv/' \
#     --ckp_name 'med_loss_prompt111111' \
#     --resume \
#     --critn 'med' \
#     >logs/med_loss_prompt111111111.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python -u finetune.py \
    --mode 'adaptor' \
    --num_epochs 50 \
    --interval 5 \
    --ckp_dir 'work_dirs/sam_psv/' \
    --ckp_name 'med_loss_adaptor' \
    --critn 'med' \
    >logs/med_loss_adaptor.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=2
# nohup python -u finetune.py \
#     --ckp_dir 'work_dirs/sam_psv/' \
#     --ckp_name 'mse_loss' \
#     --resume \
#     --critn 'mse' \
#     >logs/mse_loss.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=7
# nohup python -u finetune.py \
#     --ckp_dir 'work_dirs/sam_psv/' \
#     --ckp_name 'sam_loss' \
#     --resume \
#     --critn 'sam' \
#     >logs/sam_loss.log 2>&1 &
