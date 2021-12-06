import torch
from torch.utils.data import DataLoader

import numpy as np
import scipy.io as sio

from .utils import TedataLoader, get_PSNR, get_SSIM, inverse_gat, gat, normalize_after_gat_torch
from .unet import est_UNet

import time

torch.backends.cudnn.benchmark=True

class Test_PGE(object):
    def __init__(self,_te_data_dir=None,_pge_weight_dir=None,_save_file_name = None, _args = None):
        
        self.args = _args
        
        self.te_data_loader = TedataLoader(_te_data_dir, self.args)
        self.te_data_loader = DataLoader(self.te_data_loader, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

        self.result_alpha_arr = []
        self.result_beta_arr = []
        self.result_time_arr = []
        self.save_file_name = _save_file_name

        ## load PGE model
        
        num_output_channel = 2
        self.pge_model=est_UNet(num_output_channel,depth=3)
        self.pge_model.load_state_dict(torch.load(_pge_weight_dir))
        self.pge_model.cuda()
        
    def eval(self):
        """Evaluates denoiser on validation set."""

        alpha_arr = []
        beta_arr = []
        time_arr = []

        with torch.no_grad():

            for batch_idx, (source, target) in enumerate(self.te_data_loader):

                start = time.time()

                source = source.cuda()
                target = target.cuda()
                
                # Denoise
                est_param=self.pge_model(source)
                original_alpha=torch.mean(est_param[:,0])
                original_beta=torch.mean(est_param[:,1])

                original_beta=original_beta.cpu().numpy()
                original_alpha=original_alpha.cpu().numpy()
                
                inference_time = time.time()-start
                
                alpha_arr.append(original_alpha)
                beta_arr.append(original_beta)
                time_arr.append(inference_time)

        mean_alpha = np.mean(alpha_arr)
        mean_beta = np.mean(beta_arr)
        mean_time = np.mean(time_arr)

        print ('Mean(alpha) : ', round(mean_alpha,4), 'Mean(beta) : ', round(mean_beta,6))
            
        return 
  


