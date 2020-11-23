#! /bin/bash
# train PGE Net for a synthetic Poisson-Gaussian noise on a specific noise (alpha = 0.01, beta = 0.02)
CUDA_VISIBLE_DEVICES=3 python main.py --date 201104 --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'EMSE_Affine' --model-type 'FBI_Net' --data-type 'RawRGB' --data-name 'fivek' --alpha 0.01 --beta 0.02 --batch-size 1 --lr 0.001 --num-layers 17 --num-filters 64 --crop-size 220 --pge-weight-dir  '201104_Noise_est_RawRGB_fivek_alpha_0.01_beta_0.02_PGE_Net_cropsize_200.w' 

# # # train PGE Net for a synthetic Poisson-Gaussian noise on a random noise
# CUDA_VISIBLE_DEVICES=3 python main.py --date 201104 --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'EMSE_Affine' --model-type 'FBI_Net' --data-type 'RawRGB' --data-name 'fivek' --alpha 0.0 --beta 0.0 --batch-size 1 --lr 0.001 --num-layers 17 --num-filters 64 --crop-size 220 --pge-weight-dir 'dir_for_pge_net' 

# # # train PGE Net for FMD dataset
# 'CF_FISH', 'CF_MICE', 'TP_MICE' 
# CUDA_VISIBLE_DEVICES=3 python main.py --date 201104 --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'EMSE_Affine' --model-type 'FBI_Net' --data-type 'Grayscale' --data-name 'CF_FISH' --batch-size 1 --lr 0.001 --num-layers 17 --num-filters 64 --crop-size 220 --pge-weight-dir 'dir_for_pge_net' 

# # train PGE Net for SIDD dataset
# CUDA_VISIBLE_DEVICES=3 python main.py --date 201104 --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'EMSE_Affine' --model-type 'FBI_Net' --data-type 'RawRGB' --data-name 'SIDD' --batch-size 1 --lr 0.001 --num-layers 17 --num-filters 64 --crop-size 220 --pge-weight-dir 'SIDD_pge_est_50.w' 

# # # train PGE Net for DND dataset
# CUDA_VISIBLE_DEVICES=3 python main.py --date 201104 --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'EMSE_Affine' --model-type 'FBI_Net' --data-type 'RawRGB'--data-name 'DND' --batch-size 1 --lr 0.001 --num-layers 17 --num-filters 64 --crop-size 220 --pge-weight-dir 'dir_for_pge_net' 