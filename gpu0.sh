#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py --date 201008 --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'MSE' --model-type 'final' --data-type 'Grayscale'
CUDA_VISIBLE_DEVICES=0 python main.py --date 201008 --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'MSE' --model-type 'case1' --data-type 'Grayscale'
CUDA_VISIBLE_DEVICES=0 python main.py --date 201008 --seed 0 --noise-type 'Poisson-Gaussian' --loss-function 'MSE' --model-type 'case2' --data-type 'Grayscale'