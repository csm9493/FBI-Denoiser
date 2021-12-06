import torch
from torch.utils.data import DataLoader

import numpy as np
import scipy.io as sio

from .utils import TedataLoader, get_PSNR, get_SSIM, inverse_gat, gat, normalize_after_gat_torch
from .models import New_model
from .unet import est_UNet

import time

torch.backends.cudnn.benchmark=True

class Test_FBI(object):
    def __init__(self,_te_data_dir=None,_pge_weight_dir=None,_fbi_weight_dir=None, _save_file_name = None, _args = None):
        
        self.args = _args
        
        self.te_data_loader = TedataLoader(_te_data_dir, self.args)
        self.te_data_loader = DataLoader(self.te_data_loader, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

        self.result_psnr_arr = []
        self.result_ssim_arr = []
        self.result_time_arr = []
        self.result_denoised_img_arr = []
        self.best_psnr = 0
        self.save_file_name = _save_file_name

        num_output_channel = 2
            
        self.model = New_model(channel = 1, output_channel =  num_output_channel, filters = self.args.num_filters, num_of_layers=self.args.num_layers, case = self.args.model_type, output_type = self.args.output_type, sigmoid_value = self.args.sigmoid_value)
        self.model.load_state_dict(torch.load(_fbi_weight_dir))
        self.model.cuda()
            
        pytorch_total_params = sum([p.numel() for p in self.model.parameters()])
        print ('num of parameters : ', pytorch_total_params)
        
        ## load PGE model
        
        self.pge_model=est_UNet(num_output_channel,depth=3)
        self.pge_model.load_state_dict(torch.load(_pge_weight_dir))
        self.pge_model.cuda()
        
    def get_X_hat(self, Z, output):

        X_hat = output[:,:1] * Z + output[:,1:]
            
        return X_hat
        
    def eval(self):
        """Evaluates denoiser on validation set."""

        psnr_arr = []
        ssim_arr = []
        time_arr = []
        denoised_img_arr = []

        with torch.no_grad():

            for batch_idx, (source, target) in enumerate(self.te_data_loader):

                start = time.time()

                source = source.cuda()
                target = target.cuda()
                
                # Denoise
                est_param=self.pge_model(source)
                original_alpha=torch.mean(est_param[:,0])
                original_sigma=torch.mean(est_param[:,1])

                transformed=gat(source,original_sigma,original_alpha,0)
                transformed, transformed_sigma, min_t, max_t= normalize_after_gat_torch(transformed)

                transformed_target = torch.cat([transformed, transformed_sigma], dim = 1)
#                     target = torch.cat([target,transformed_sigma], dim = 1)

                output = self.model(transformed)

                transformed_Z = transformed_target[:,:1]
                X = target.cpu().numpy()
                X_hat = self.get_X_hat(transformed_Z,output).cpu().numpy()

                transformed=transformed.cpu().numpy()
                original_sigma=original_sigma.cpu().numpy()
                original_alpha=original_alpha.cpu().numpy()
                
                min_t=min_t.cpu().numpy()
                max_t=max_t.cpu().numpy()
                
                X_hat =X_hat*(max_t-min_t)+min_t
                X_hat=np.clip(inverse_gat(X_hat,original_sigma,original_alpha,0,method='closed_form'), 0, 1)
                
                inference_time = time.time()-start
                
                psnr_arr.append(get_PSNR(X[0], X_hat[0]))
                ssim_arr.append(get_SSIM(X[0], X_hat[0]))
                time_arr.append(inference_time)
                denoised_img_arr.append(X_hat[0].reshape(X_hat.shape[2],X_hat.shape[3]))

        mean_psnr = np.mean(psnr_arr)
        mean_ssim = np.mean(ssim_arr)
        mean_time = np.mean(time_arr)

        if self.best_psnr <= mean_psnr:
            self.best_psnr = mean_psnr
            self.result_denoised_img_arr = denoised_img_arr.copy()
            
        print ('PSNR : ', round(mean_psnr,4), '\tSSIM : ', round(mean_ssim,4))
            
        return 
  



