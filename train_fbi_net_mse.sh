#! /bin/bash
# train PGE Net for a synthetic Poisson-Gaussian noise on a specific noise (alpha = 0.01, beta = 0.02)
CUDA_VISIBLE_DEVICES=0 python main.py --date 201104 --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'MSE_Affine' --model-type 'FBI_Net' --data-type 'RawRGB' --data-name 'fivek' --alpha 0.01 --beta 0.02 --batch-size 1 --lr 0.001 --num-layers 17 --num-filters 64 --crop-size 220 

# train PGE Net for a synthetic Poisson-Gaussian noise on a mixture noise
CUDA_VISIBLE_DEVICES=0 python main.py --date 201104 --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'MSE_Affine' --model-type 'FBI_Net' --data-type 'RawRGB' --data-name 'fivek' --alpha 0.0 --beta 0.0 --batch-size 1 --lr 0.001 --num-layers 17 --num-filters 64 --crop-size 220 