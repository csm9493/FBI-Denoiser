import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Denoising')
    # Arguments
    parser.add_argument('--date', type=str, default='', help='(default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--noise-type', default='Real', type=str, required=False,
                        choices=['Poisson-Gaussian'],
                        help='(default=%(default)s)')
    parser.add_argument('--data-type', default='RawRGB', type=str, required=False,
                        choices=['Grayscale',
                                 'RawRGB',],
                        help='(default=%(default)s)')
    parser.add_argument('--data-name', default='BSD', type=str, required=False,
                        choices=['BSD',
                                 'fivek'],
                        help='(default=%(default)s)')
    
    parser.add_argument('--alpha', default=0.01, type=float, help='(default=%(default)f)')
    parser.add_argument('--beta', default=0.02, type=float, help='(default=%(default)f)')
    
    args=parser.parse_args()
    return args



