#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python custom_main.py --date 201102 --seed 0 --noise-type 'Poisson-Gaussian' --data-type 'RawRGB' --data-name 'fivek' --alpha 0.01 --beta 0.02

# CUDA_VISIBLE_DEVICES=0 python 200920_N2V_Medical_datset2_Dose10_Unet.py
# CUDA_VISIBLE_DEVICES=0 python 200920_N2V_Medical_datset2_Dose15_Unet.py
# CUDA_VISIBLE_DEVICES=0 python 200920_N2V_Medical_datset2_Dose20_Unet.py
# CUDA_VISIBLE_DEVICES=0 python 200920_N2V_SIDD_RawRGB_Unet.py

# CUDA_VISIBLE_DEVICES=0 python N2V_RealFM_TP.py
# CUDA_VISIBLE_DEVICES=0 python N2V_SIDD_rawRGB.py