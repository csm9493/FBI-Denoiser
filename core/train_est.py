import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torchvision.transforms.functional as tvF
import scipy.interpolate as sip
from .utils import TrdataLoader, TedataLoader, get_PSNR, get_SSIM, chen_estimate, gat
# from .loss_functions import *
from .logger import Logger
import torchvision as vision
import sys
from .unet import est_UNet  
import time

class Train_Est(object):
    def __init__(self,_tr_data_dir=None, _te_data_dir=None, _save_file_name = None, _args = None):
        
        self.tr_data_dir = _tr_data_dir
        self.te_data_dir = _te_data_dir
        self.args = _args
        self.save_file_name = _save_file_name
        
        self.tr_data_loader = TrdataLoader(_tr_data_dir, self.args)
        self.tr_data_loader = DataLoader(self.tr_data_loader, batch_size=self.args.batch_size, shuffle=True, num_workers=0, drop_last=True)

        self.te_data_loader = TedataLoader(_te_data_dir, self.args)
        self.te_data_loader = DataLoader(self.te_data_loader, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

        self.result_tr_loss_arr = []

        self.logger = Logger(self.args.nepochs, len(self.tr_data_loader))
        self._compile()

    def _compile(self):
        
        self.model=est_UNet(2,depth=self.args.unet_layer)
        
        pytorch_total_params = sum([p.numel() for p in self.model.parameters()])
        print ('num of parameters : ', pytorch_total_params)
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, self.args.drop_epoch, gamma=self.args.drop_rate)
        
        self.model = self.model.cuda()
 
       
    def save_model(self, epoch):
        torch.save(self.model.state_dict(), './weights/'+self.save_file_name  +'_ep'+ str(epoch) + '.w')
        return
    
        
    def eval(self):
        """Evaluates denoiser on validation set."""        
        
        a_arr=[]
        b_arr=[]

        with torch.no_grad():
            for batch_idx, (source, target) in enumerate(self.te_data_loader):
                
                source = source.cuda()
                target = target.cuda()
                # Denoise
                
                output = self.model(source)
  
                target = target.cpu().numpy()
                output = output.cpu().numpy()

                a_arr.append(np.mean(output[:,0]))
                b_arr.append(np.mean(output[:,1]))

        return a_arr,b_arr 
    
    def _on_epoch_end(self, epoch, mean_tr_loss):
        """Tracks and saves starts after each epoch."""
        self.save_model(epoch)
        a_arr, b_arr = self.eval()
        self.result_tr_loss_arr.append(mean_tr_loss)
        sio.savemat('./result_data/'+self.save_file_name +'_result',{'tr_loss_arr':self.result_tr_loss_arr,'a_arr':a_arr, 'b_arr':b_arr})
            
    def _vst(self,transformed):    
        
        est=chen_estimate(transformed)
        print (est)
#         print('chen est: ' ,est)
        return ((est-1)**2)
     
    def train(self):
        """Trains denoiser on training set."""
        num_batches = len(self.tr_data_loader)
      
        for epoch in range(self.args.nepochs):
            self.scheduler.step()
            tr_loss = []

            for batch_idx, (source, target) in enumerate(self.tr_data_loader):
                self.optim.zero_grad()
                source = source.cuda()
                target = target.cuda()
                
                noise_hat=self.model(source)
                
                predict_alpha=torch.mean(noise_hat[:,0])
                predict_sigma=torch.mean(noise_hat[:,1])
                
                predict_gat=gat(source,predict_sigma,predict_alpha,0)     
#                 predict_gat=gat(source,torch.tensor(0.02).to(torch.float32),torch.tensor(0.01).to(torch.float32),0)     
                
                loss=self._vst(predict_gat)

                self.logger.log(losses = {'loss': loss, 'pred_alpha': predict_alpha, 'pred_sigma': predict_sigma}, lr = self.optim.param_groups[0]['lr'])
                tr_loss.append(loss.detach().cpu().numpy())

            mean_tr_loss = np.mean(tr_loss)
            self._on_epoch_end(epoch+1, mean_tr_loss)    
            



