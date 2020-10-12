from core.train import Train
from arguments import get_args
import torch
import numpy as np
import random

args = get_args()

# control the randomness
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

if __name__ == '__main__':
    """Trains Noise2Noise."""
    
    if args.noise_type == 'Poisson-Gaussian':
        
        if args.data_type == 'Grayscale':

            tr_data_dir = './data/train_BSD300_grayscale_25000x256x256_cropped_alpha_'+str(args.alpha)+'_beta_'+str(args.beta)+'.hdf5'
            te_data_dir = './data/test_BSD68_grayscale_alpha_'+str(args.alpha)+'_beta_'+str(args.beta)+'.hdf5'
        else:
            
            tr_data_dir = './data/train_fivek_rawRGB_25000x256x256_cropped_alpha_'+str(args.alpha)+'_beta_'+str(args.beta)+'.hdf5'
            te_data_dir = './data/test_fivek_rawRGB_alpha_'+str(args.alpha)+'_beta_'+str(args.beta)+'.hdf5'
    
    if args.model_type == 'FC-AIDE':
        save_file_name = str(args.date)+ '_' + str(args.loss_function) + '_' + str(args.data_type) +'_'+ str(args.data_name)+ '_alpha_' + str(args.alpha) + '_beta_' + str(args.beta) + '_' + str(args.model_type) + '_layers_x' + str(10) + '_filters_x' + str(64)
    else:
        save_file_name = str(args.date)+ '_' + str(args.loss_function) + '_' + str(args.data_type) +'_'+ str(args.data_name)+ '_alpha_' + str(args.alpha) + '_beta_' + str(args.beta) + '_' + str(args.model_type) + '_layers_x' + str(args.num_layers) + '_filters_x' + str(args.num_filters)
    
    print ('save_file_name : ', save_file_name)
    
    # Initialize model and train
    train = Train(_tr_data_dir=tr_data_dir, _te_data_dir=te_data_dir, _save_file_name = save_file_name,  _args = args)
    train.train()