from core.train_fbi import Train_FBI
from core.train_pge import Train_PGE
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
        
        if args.data_type == 'RawRGB' and args.data_name == 'SIDD':

            tr_data_dir = './data/train_SIDD_25000_2.hdf5'
            te_data_dir = './data/test_SIDD.hdf5'
            
            save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_'+ str(args.data_name)
            
        elif args.data_type == 'RawRGB' and args.data_name == 'DND':

            tr_data_dir = './data/train_DND_25000_2.hdf5'
            te_data_dir = './data/test_SIDD.hdf5'
            
            save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_'+ str(args.data_name)
            
        elif args.data_type == 'Grayscale' and args.data_name == 'CF_FISH':

            tr_data_dir = './data/train_CF_FISH_25000x256x256_2.hdf5'
            te_data_dir = './data/test_CF_FISH_raw.hdf5'
            
            save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_'+ str(args.data_name)
            
        elif args.data_type == 'Grayscale' and args.data_name == 'CF_MICE':

            tr_data_dir = './data/train_CF_MICE_25000x256x256_2.hdf5'
            te_data_dir = './data/test_CF_MICE_raw.hdf5'
            
            save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_'+ str(args.data_name)
            
        elif args.data_type == 'Grayscale' and args.data_name == 'TP_MICE':

            tr_data_dir = './data/train_TP_MICE_25000x256x256_2.hdf5'
            te_data_dir = './data/test_TP_MICE_raw.hdf5'
            
            save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_'+ str(args.data_name)
            
        elif args.data_type == 'RawRGB' and args.data_name == 'fivek' and args.alpha == 0 and args.beta == 0:
            
            tr_data_dir = './data/train_fivek_rawRGB_25000x256x256_cropped_random_noise.hdf5'
            te_data_dir = './data/test_fivek_rawRGB_random_noise.hdf5'
            
            save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_' + 'random_noise'
        
        else:
            
            tr_data_dir = './data/train_fivek_rawRGB_25000x256x256_cropped_alpha_'+str(args.alpha)+'_beta_'+str(args.beta)+'.hdf5'
            te_data_dir = './data/test_fivek_rawRGB_alpha_'+str(args.alpha)+'_beta_'+str(args.beta)+'.hdf5'
            
            save_file_name = str(args.date)+ '_'+str(args.model_type)+'_' + str(args.data_type) +'_'+ str(args.data_name)+ '_alpha_' + str(args.alpha) + '_beta_' + str(args.beta)
            

        print ('tr data dir : ', tr_data_dir)
        print ('te data dir : ', te_data_dir)
    
    if args.model_type == 'FC-AIDE':
        save_file_name += '_layers_x' + str(10) + '_filters_x' + str(64)
    elif args.model_type == 'DBSN':
        save_file_name = ''
    elif args.model_type == 'PGE_Net':
        save_file_name += '_cropsize_' + str(args.crop_size)
    elif args.model_type == 'FBI_Net':
        save_file_name += '_layers_x' + str(args.num_layers) + '_filters_x' + str(args.num_filters)+ '_cropsize_' + str(args.crop_size)
    
    print ('save_file_name : ', save_file_name)
    
    # Initialize model and train
    if args.model_type != 'PGE_Net':
        train = Train_FBI(_tr_data_dir=tr_data_dir, _te_data_dir=te_data_dir, _save_file_name = save_file_name,  _args = args)
    else:
        train = Train_PGE(_tr_data_dir=tr_data_dir, _te_data_dir=te_data_dir, _save_file_name = save_file_name,  _args = args)
    train.train()