import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Denoising')
    # Arguments
    parser.add_argument('--date', type=str, default='', help='(default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--noise-type', default='Real', type=str, required=False,
                        choices=['Poisson-Gaussian'],
                        help='(default=%(default)s)')
    parser.add_argument('--loss-function', default='Estimated_Affine', type=str, required=False,
                        choices=['MSE', 'N2V', 'MSE_Affine', 'Noise_est', 'EMSE_Affine'],
                        help='(default=%(default)s)')
    parser.add_argument('--model-type', default='final', type=str, required=False,
                        choices=['case1',
                                 'case2',
                                 'case3',
                                 'case4',
                                 'case5',
                                 'case6',
                                 'case7',
                                 'FBI_Net',
                                 'PGE_Net',
                                 'DBSN',
                                 'FC-AIDE'],
                        help='(default=%(default)s)')
    parser.add_argument('--data-type', default='RawRGB', type=str, required=False,
                        choices=['Grayscale',
                                 'RawRGB',],
                        help='(default=%(default)s)')
    parser.add_argument('--data-name', default='BSD', type=str, required=False,
                        choices=['BSD',
                                 'fivek',
                                 'SIDD',
                                 'DND',
                                 'CF_FISH',
                                 'CF_MICE',
                                 'TP_MICE'],
                        help='(default=%(default)s)')
    
    parser.add_argument('--nepochs', default=50, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--batch-size', default=4, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--lr', default=0.001, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--drop-rate', default=0.5, type=float, help='(default=%(default)f)')
    parser.add_argument('--drop-epoch', default=10, type=int, help='(default=%(default)f)')
    parser.add_argument('--crop-size', default=120, type=int, help='(default=%(default)f)')
    
    parser.add_argument('--alpha', default=0.01, type=float, help='(default=%(default)f)')
    parser.add_argument('--beta', default=0.02, type=float, help='(default=%(default)f)')
    
    parser.add_argument('--num-layers', default=8, type=int, help='(default=%(default)f)')
    parser.add_argument('--num-filters', default=64, type=int, help='(default=%(default)f)')
    parser.add_argument('--mul', default=1, type=int, help='(default=%(default)f)')
    
    
    parser.add_argument('--unet-layer', default=3, type=int, help='(default=%(default)f)')
    parser.add_argument('--pge-weight-dir', default=None, type=str, help='(default=%(default)f)')
    
    parser.add_argument('--output-type', default='sigmoid', type=str, help='(default=%(default)f)')
    parser.add_argument('--sigmoid-value', default=0.1, type=float, help='(default=%(default)f)')
    
    
    args=parser.parse_args()
    return args




