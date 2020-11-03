#! /bin/bash

python main.py --date 201102 --seed 0 --noise-type 'Poisson-Gaussian' --data-type 'RawRGB' --data-name 'fivek' --alpha 0.01 --beta 0.02
python main.py --date 201102 --seed 0 --noise-type 'Poisson-Gaussian' --data-type 'RawRGB' --data-name 'fivek' --alpha 0.01 --beta 0.0002
python main.py --date 201102 --seed 0 --noise-type 'Poisson-Gaussian' --data-type 'RawRGB' --data-name 'fivek' --alpha 0.0005 --beta 0.02
python main.py --date 201102 --seed 0 --noise-type 'Poisson-Gaussian' --data-type 'RawRGB' --data-name 'fivek' --alpha 0.0005 --beta 0.0002
