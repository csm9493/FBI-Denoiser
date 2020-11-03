import bm3d
import scipy.io as sio
import numpy as np
from sklearn.metrics import mean_squared_error
from skimage import measure
import math
from cv2 import imread
from skimage import img_as_float
import time
from arguments import get_args
import h5py

args = get_args()

def get_PSNR(X, X_hat):

    mse = mean_squared_error(X,X_hat)
    test_PSNR = 10 * math.log10(1/mse)

    return test_PSNR

def get_SSIM(X, X_hat):

    test_SSIM = measure.compare_ssim(X, X_hat, data_range=X.max() - X.min())

    return test_SSIM

def im2patch(im, pch_size, stride=1):
    '''
    Transform image to patches.
    Input:
        im: 3 x H x W or 1 X H x W image, numpy format
        pch_size: (int, int) tuple or integer
        stride: (int, int) tuple or integer
    '''
    if isinstance(pch_size, tuple):
        pch_H, pch_W = pch_size
    elif isinstance(pch_size, int):
        pch_H = pch_W = pch_size
    else:
        sys.exit('The input of pch_size must be a integer or a int tuple!')

    if isinstance(stride, tuple):
        stride_H, stride_W = stride
    elif isinstance(stride, int):
        stride_H = stride_W = stride
    else:
        sys.exit('The input of stride must be a integer or a int tuple!')


    C, H, W = im.shape
    num_H = len(range(0, H-pch_H+1, stride_H))
    num_W = len(range(0, W-pch_W+1, stride_W))
    num_pch = num_H * num_W
    pch = np.zeros((C, pch_H*pch_W, num_pch), dtype=im.dtype)
    kk = 0
    for ii in range(pch_H):
        for jj in range(pch_W):
            temp = im[:, ii:H-pch_H+ii+1:stride_H, jj:W-pch_W+jj+1:stride_W]
            pch[:, kk, :] = temp.reshape((C, num_pch))
            kk += 1

    return pch.reshape((C, pch_H, pch_W, num_pch))

def noise_estimate(im, pch_size=8):
    '''
    Implement of noise level estimation of the following paper:
    Chen G , Zhu F , Heng P A . An Efficient Statistical Method for Image Noise Level Estimation[C]// 2015 IEEE International Conference
    on Computer Vision (ICCV). IEEE Computer Society, 2015.
    Input:
        im: the noise image, H x W x 3 or H x W numpy tensor, range [0,1]
        pch_size: patch_size
    Output:
        noise_level: the estimated noise level
    '''

    if im.ndim == 3:
        im = im.transpose((2, 0, 1))
    else:
        im = np.expand_dims(im, axis=0)

    # image to patch
    pch = im2patch(im, pch_size, 3)  # C x pch_size x pch_size x num_pch tensor
    num_pch = pch.shape[3]
    pch = pch.reshape((-1, num_pch))  # d x num_pch matrix
    d = pch.shape[0]

    mu = pch.mean(axis=1, keepdims=True)  # d x 1
    X = pch - mu
    sigma_X = np.matmul(X, X.transpose()) / num_pch
    sig_value, _ = np.linalg.eigh(sigma_X)
    sig_value.sort()

    for ii in range(-1, -d-1, -1):
        tau = np.mean(sig_value[:ii])
        if np.sum(sig_value[:ii]>tau) == np.sum(sig_value[:ii] < tau):
            return np.sqrt(tau)
        
        
def generalized_anscombe(x, mu, sigma, gain=1.0):
    '''
    Compute the generalized anscombe variance stabilizing transform,
    which assumes that the data provided to it is a mixture of poisson
    and gaussian noise.
    The input signal  z  is assumed to follow the Poisson-Gaussian noise model
        x = gain * p + n
    where gain is the camera gain and mu and sigma are the read noise
    mean and standard deviation.
    We assume that x contains only positive values.  Values that are
    less than or equal to 0 are ignored by the transform.
    Note, this transform will show some bias for counts less than
    about 20.
    '''
    y = gain*x + (gain**2)*3.0/8.0 + sigma**2 - gain*mu

    # Clamp to zero before taking the square root.
    return (2.0/gain)*np.sqrt(np.maximum(y, 0.0))

def inverse_generalized_anscombe(x, mu, sigma, gain=1.0):
    '''
    Applies the closed-form approximation of the exact unbiased
    inverse of Generalized Anscombe variance-stabilizing
    transformation.
    The input signal x is transform back into a Poisson random variable
    based on the assumption that the original signal from which it was
    derived follows the Poisson-Gaussian noise model:
        x = gain * p + n
    where gain is the camera gain and mu and sigma are the read noise
    mean and standard deviation.
    Roference: M. Makitalo and A. Foi, "Optimal inversion of the
    generalized Anscombe transformation for Poisson-Gaussian noise",
    IEEE Trans. Image Process., doi:10.1109/TIP.2012.2202675
    '''
    test = np.maximum(x, 1.0)
    exact_inverse = ( np.power(test/2.0, 2.0) +
                      1.0/4.0 * np.sqrt(3.0/2.0)*np.power(test, -1.0) -
                      11.0/8.0 * np.power(test, -2.0) +
                      5.0/8.0 * np.sqrt(3.0/2.0) * np.power(test, -3.0) -
                      1.0/8.0 - np.power(sigma, 2) )
    exact_inverse = np.maximum(0.0, exact_inverse)
    exact_inverse *= gain
    exact_inverse += mu
    exact_inverse[np.where(exact_inverse != exact_inverse)] = 0.0
    return exact_inverse

if __name__ == '__main__':
    """Trains Noise2Noise."""
    
    if args.noise_type == 'Poisson-Gaussian':
        
        if args.data_type == 'Grayscale':

            tr_data_dir = '../../data/train_BSD300_grayscale_25000x256x256_cropped_alpha_'+str(args.alpha)+'_beta_'+str(args.beta)+'.hdf5'
            te_data_dir = '../../data/test_BSD68_grayscale_alpha_'+str(args.alpha)+'_beta_'+str(args.beta)+'.hdf5'
        else:
            
            tr_data_dir = '../../data/train_fivek_rawRGB_25000x256x256_cropped_alpha_'+str(args.alpha)+'_beta_'+str(args.beta)+'.hdf5'
            te_data_dir = '../../data/test_fivek_rawRGB_alpha_'+str(args.alpha)+'_beta_'+str(args.beta)+'.hdf5'
            
        print ('tr data dir : ', tr_data_dir)
        print ('te data dir : ', te_data_dir)
    
        save_file_name = str(args.date)+ '_BM3D_' + str(args.data_type) +'_'+ str(args.data_name)+ '_alpha_' + str(args.alpha) + '_beta_' + str(args.beta)
    
    print ('save_file_name : ', save_file_name)

    psnr_arr = []
    ssim_arr = []
    denoised_img_arr = []
    time_arr = []

    data = h5py.File(te_data_dir, "r")
    clean_images = data["clean_images"][:,:,:]
    noisy_images = data["noisy_images"][:,:,:]

    for idx in range(noisy_images.shape[0]):

        start = time.time()
        noisy_GAT = generalized_anscombe(noisy_images[idx], 0, args.beta, args.alpha)
        est_level = noise_estimate(noisy_GAT, 8)
        denoised_img = np.clip(inverse_generalized_anscombe(bm3d.bm3d(noisy_GAT, sigma_psd=est_level, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING), 0, args.beta, args.alpha), 0, 1)
        end = time.time()
        time_elapsed = end -start

        psnr_arr.append(get_PSNR(clean_images[idx],denoised_img))
        ssim_arr.append(get_SSIM(clean_images[idx],denoised_img))
        denoised_img_arr.append(denoised_img)
        time_arr.append(time_elapsed)

    print (Dose_arr[i], ' : ', 'PSNR : ', np.mean(psnr_arr), ' SSIM : ', np.mean(ssim_arr), ' Time : ', np.mean(time_arr))

    sio.savemat('../../result_data/'+ save_file_name, {'psnr_arr':np.array(psnr_arr), 'ssim_arr':np.array(ssim_arr), 'denoised_img_arr':np.array(denoised_img_arr), 'time_arr':np.array(time_arr) })






