#
import os
import random
import datetime
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from data import create_dataset
import scipy.io as sio
import h5py

from gray_options import opt
from net.backbone_net import DBSN_Model
from net.sigma_net import Sigma_mu_Net, Sigma_n_Net
# from util.utils import batch_psnr

seed=0
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if seed == 0:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def load_nlf(info, img_id):
    nlf = {}
    nlf_h5 = info[info["nlf"][0][img_id]]
    nlf["a"] = nlf_h5["a"][0][0]
    nlf["b"] = nlf_h5["b"][0][0]
    return nlf

def load_sigma_raw(info, img_id, bb, yy, xx):
    nlf_h5 = info[info["sigma_raw"][0][img_id]]
    sigma = nlf_h5[xx,yy,bb]
    return sigma

def main(args):
    # net architecture
    dbsn_net = DBSN_Model(in_ch = args.input_channel,
                            out_ch = args.output_channel,
                            mid_ch = args.middle_channel,
                            blindspot_conv_type = args.blindspot_conv_type,
                            blindspot_conv_bias = args.blindspot_conv_bias,
                            br1_block_num = args.br1_block_num,
                            br1_blindspot_conv_ks =args.br1_blindspot_conv_ks,
                            br2_block_num = args.br2_block_num,
                            br2_blindspot_conv_ks = args.br2_blindspot_conv_ks,
                            activate_fun = args.activate_fun)
    sigma_mu_net = Sigma_mu_Net(in_ch=args.middle_channel,
                    out_ch=args.sigma_mu_output_channel,
                    mid_ch=args.sigma_mu_middle_channel,
                    layers=args.sigma_mu_layers,
                    kernel_size=args.sigma_mu_kernel_size,
                    bias=args.sigma_mu_bias)
    sigma_n_net = Sigma_n_Net(in_ch=args.sigma_n_input_channel,
            out_ch=args.sigma_n_output_channel,
            mid_ch=args.sigma_n_middle_channel,
            layers=args.sigma_n_layers,
            kernel_size=args.sigma_n_kernel_size,
            bias=args.sigma_n_bias)

    # Move to GPU
    dbsn_model = nn.DataParallel(dbsn_net, args.device_ids).cuda()
    sigma_mu_model = nn.DataParallel(sigma_mu_net, args.device_ids).cuda()
    sigma_n_model = nn.DataParallel(sigma_n_net, args.device_ids).cuda()

#     tmp_ckpt=torch.load(args.last_ckpt,map_location=torch.device('cuda', args.device_ids[0]))
#     # Initialize dbsn_model
#     pretrained_dict=tmp_ckpt['state_dict_dbsn']
#     model_dict=dbsn_model.state_dict()
#     pretrained_dict_update = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     assert(len(pretrained_dict)==len(pretrained_dict_update))
#     assert(len(pretrained_dict_update)==len(model_dict))
#     model_dict.update(pretrained_dict_update)
    dbsn_model.load_state_dict(torch.load('../../../weights/201104_DBSN_RawRGB_SIDD_dbsn_model.w'))

    # Initialize sigma_mu_model
#     pretrained_dict=tmp_ckpt['state_dict_sigma_mu']
#     model_dict=sigma_mu_model.state_dict()
#     pretrained_dict_update = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     assert(len(pretrained_dict)==len(pretrained_dict_update))
#     assert(len(pretrained_dict_update)==len(model_dict))
#     model_dict.update(pretrained_dict_update)
    sigma_mu_model.load_state_dict(torch.load('../../../weights/201104_DBSN_RawRGB_SIDD_sigma_mu_model.w'))

    # Initialize sigma_n_model
#     pretrained_dict=tmp_ckpt['state_dict_sigma_n']
#     model_dict=sigma_n_model.state_dict()
#     pretrained_dict_update = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     assert(len(pretrained_dict)==len(pretrained_dict_update))
#     assert(len(pretrained_dict_update)==len(model_dict))
#     model_dict.update(pretrained_dict_update)
    sigma_n_model.load_state_dict(torch.load('../../../weights/201104_DBSN_RawRGB_SIDD_sigma_n_model.w'))

#     # set val set
#     val_setname = args.valset
#     dataset_val = create_dataset(val_setname, 'val', args).load_data()

#     # --------------------------------------------
#     # Evaluation
#     # --------------------------------------------
#     print("Evaluation on : %s " % (val_setname))


    dbsn_model.eval()
    sigma_mu_model.eval()
    sigma_n_model.eval()
    
    '''
    Utility function for denoising all bounding boxes in all raw images of
    the DND dataset.

    denoiser      Function handle
                  It is called as Idenoised = denoiser(Inoisy, nlf) where Inoisy is the noisy image patch 
                  and nlf is a dictionary containing the parameters of the noise level
                  function (nlf["a"], nlf["b"]) and a mean noise strength (nlf["sigma"])
    data_folder   Folder where the DND dataset resides
    out_folder    Folder where denoised output should be written to
    '''
    
    # load info
    
    data_folder = '../../../data/DND/dnd_2017_split/'
    out_folder_xhat = '../../../result_data/DBSN_xhat_DND_benchmark/'
    out_folder_mu = '../../../result_data/DBSN_mu_DND_benchmark/'
    
    try:
        os.makedirs(out_folder_xhat)
        os.makedirs(out_folder_mu)
    except:pass
    
    infos = h5py.File(os.path.join(data_folder, 'info.mat'), 'r')
    info = infos['info']
    bb = info['boundingboxes']
    print('info loaded\n')
    # process data
    for i in range(50):
        filename = os.path.join(data_folder, 'images_raw', '%04d.mat'%(i+1))
        img = h5py.File(filename, 'r')
        Inoisy = np.float32(np.array(img['Inoisy']).T)
        # bounding box
        ref = bb[0][i]
        H=256
        W=256

        with torch.no_grad():
            boxes = np.array(info[ref]).T
            for k in range(20):
                idx = [int(boxes[k,0]-1),int(boxes[k,2]),int(boxes[k,1]-1),int(boxes[k,3])]
                Inoisy_crop = Inoisy[idx[0]:idx[1],idx[2]:idx[3]].copy()
                Idenoised_crop_xhat = Inoisy_crop.copy()
                Idenoised_crop_mu = Inoisy_crop.copy()
         #       H = Inoisy_crop.shape[0]
         #       W = Inoisy_crop.shape[1]
                nlf = load_nlf(info, i)

    #########################################################################################3
                for yy in range(2):
                    for xx in range(2):
                        nlf["sigma"] = load_sigma_raw(info, i, k, yy, xx)
                        img_noise_val = Inoisy_crop[yy*H:yy*H+H, xx*W:xx*W+W].copy()
                        img_noise_val = torch.from_numpy(img_noise_val).cuda()
                        img_noise_val = img_noise_val.reshape(1,1,img_noise_val.shape[0],img_noise_val.shape[1])
    #                     Idenoised_crop_c = denoiser(Inoisy_crop_c, nlf)

                        mu_out_val, mid_out_val = dbsn_model(img_noise_val)
                        # forward sigma_mu
                        sigma_mu_out_val = sigma_mu_model(mid_out_val)
                        sigma_mu_val = sigma_mu_out_val ** 2
                        # forward sigma_n
                        sigma_n_out_val = sigma_n_model(mu_out_val)
                        noise_est_val = F.softplus(sigma_n_out_val - 4) + (1e-3)
                        sigma_n_val = noise_est_val ** 2
                        # MAP inference
                        map_out_val = (img_noise_val * sigma_mu_val +args.gamma* mu_out_val * sigma_n_val) / (sigma_mu_val + args.gamma*sigma_n_val)

                        Idenoised_crop_xhat[yy*H:yy*H+H,xx*W:xx*W+W] = map_out_val.detach().cpu().numpy()
                        Idenoised_crop_mu[yy*H:yy*H+H,xx*W:xx*W+W] = mu_out_val.detach().cpu().numpy()
    ##########################################################################################
                # save denoised data
                Idenoised_crop_xhat = np.float32(Idenoised_crop_xhat)
                save_file = os.path.join(out_folder_xhat, '%04d_%02d.mat'%(i+1,k+1))
                sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop_xhat})
                
                # save denoised data
                Idenoised_crop_mu = np.float32(Idenoised_crop_mu)
                save_file = os.path.join(out_folder_mu, '%04d_%02d.mat'%(i+1,k+1))
                sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop_mu})
                print('%s crop %d/%d' % (filename, k+1, 20))
            print('[%d/%d] %s done\n' % (i+1, 50, filename))

    
if __name__ == "__main__":

    main(opt)

    exit(0)






