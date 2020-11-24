#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python custom_main.py --date 201102 --seed 0 --noise-type 'Poisson-Gaussian' --data-type 'RawRGB' --data-name 'fivek' --alpha 0.01 --beta 0.02
