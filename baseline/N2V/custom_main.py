import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from unet.model import UNet

from pn2v import utils
from pn2v import histNoiseModel
from pn2v import training_custom as training

from tifffile import imread
import h5py
from arguments import get_args

# See if we can use a GPU
device=utils.getDevice()
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

            tr_data_dir = '../../../data/train_SIDD_25000_2.hdf5'
            te_data_dir = '../../../data/test_SIDD.hdf5'
            
            save_file_name = str(args.date)+ '_N2V_' + str(args.data_type) +'_'+ str(args.data_name)
            
        elif args.data_type == 'RawRGB' and args.data_name == 'DND':

            tr_data_dir = '../../../data/train_DND_25000_2.hdf5'
            te_data_dir = '../../../data/test_SIDD.hdf5'
            
            save_file_name = str(args.date)+ '_N2V_' + str(args.data_type) +'_'+ str(args.data_name)
            
        elif args.data_type == 'Grayscale' and args.data_name == 'CF_FISH':

            tr_data_dir = '../../../data/train_CF_FISH_25000x256x256_2.hdf5'
            te_data_dir = '../../../data/test_CF_FISH_raw.hdf5'
            
            save_file_name = str(args.date)+ '_N2V_' + str(args.data_type) +'_'+ str(args.data_name)
            
        elif args.data_type == 'Grayscale' and args.data_name == 'CF_MICE':

            tr_data_dir = '../../../data/train_CF_MICE_25000x256x256_2.hdf5'
            te_data_dir = '../../../data/test_CF_MICE_raw.hdf5'
            
            save_file_name = str(args.date)+ '_N2V_' + str(args.data_type) +'_'+ str(args.data_name)
            
        elif args.data_type == 'Grayscale' and args.data_name == 'TP_MICE':

            tr_data_dir = '../../../data/train_TP_MICE_25000x256x256_2.hdf5'
            te_data_dir = '../../../data/test_TP_MICE_raw.hdf5'
            
            save_file_name = str(args.date)+ '_N2V_' + str(args.data_type) +'_'+ str(args.data_name)
            
        elif args.data_type == 'RawRGB' and args.data_name == 'fivek' and args.alpha == 0 and args.beta == 0:
            
            tr_data_dir = '../../../data/train_fivek_rawRGB_25000x256x256_cropped_random_noise.hdf5'
            te_data_dir = '../../../data/test_fivek_rawRGB_random_noise.hdf5'
            
            save_file_name = str(args.date)+ '_N2V_' + str(args.data_type) +'_' + 'random_noise'
        
        else:
            
            tr_data_dir = '../../../data/train_fivek_rawRGB_25000x256x256_cropped_alpha_'+str(args.alpha)+'_beta_'+str(args.beta)+'.hdf5'
            te_data_dir = '../../../data/test_fivek_rawRGB_alpha_'+str(args.alpha)+'_beta_'+str(args.beta)+'.hdf5'
            
            save_file_name = str(args.date)+ '_N2V_' + str(args.data_type) +'_'+ str(args.data_name)+ '_alpha_' + str(args.alpha) + '_beta_' + str(args.beta)
            
        print ('tr data dir : ', tr_data_dir)
        print ('te data dir : ', te_data_dir)
    
   
    print ('save_file_name : ', save_file_name)

# # Load the training data

data = h5py.File(tr_data_dir, "r")
tr_source = data["noisy_images"][:22000,:,:]
val_source = data["noisy_images"][22000:,:,:]

data = h5py.File(te_data_dir, "r")
te_target = data["clean_images"][:,:,:]
te_source = data["noisy_images"][:,:,:]


# tr_target = data["clean"][:,:,:]


# The N2V network requires only a single output unit per pixel
net = UNet(1, depth=3)

# Split training and validation data.
my_train_data=tr_source.copy()
my_val_data_source=te_source.copy()

# Start training.
trainHist, valHist = training.trainNetwork(net=net, trainData=tr_source, valData=val_source, te_Data_target = te_target, te_Data_source = te_source,
                                           postfix='conv_N2V', directory=None, noiseModel=None,
                                           device=device, numOfEpochs= 200, stepsPerEpoch=10, 
                                           virtualBatchSize=20, batchSize=1, learningRate=1e-3, save_file_name = save_file_name)
