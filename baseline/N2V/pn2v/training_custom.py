import torch.optim as optim
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
import matplotlib.pyplot as plt
import torchvision

from pn2v import utils
from pn2v import prediction

import math
from skimage import measure
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import image
from pn2v.utils import PSNR
import scipy.io as sio

import time

############################################
#   Training the network
############################################


def get_SSIM(X, X_hat):

    test_SSIM = measure.compare_ssim(X, X_hat, data_range=X.max() - X.min(), multichannel=True)

    return test_SSIM


def getStratifiedCoords2D(numPix, shape):
    '''
    Produce a list of approx. 'numPix' random coordinate, sampled from 'shape' using startified sampling.
    '''
    box_size = np.round(np.sqrt(shape[0] * shape[1] / numPix)).astype(np.int)
    coords = []
    box_count_y = int(np.ceil(shape[0] / box_size))
    box_count_x = int(np.ceil(shape[1] / box_size))
    for i in range(box_count_y):
        for j in range(box_count_x):
            y = np.random.randint(0, box_size)
            x = np.random.randint(0, box_size)
            y = int(i * box_size + y)
            x = int(j * box_size + x)
            if (y < shape[0] and x < shape[1]):
                coords.append((y, x))
    return coords


def randomCropFRI(data, size, numPix, supervised=False, counter=None, augment=True):
    '''
    Crop a patch from the next image in the dataset.
    The patches are augmented by randomly deciding to mirror them and/or rotating them by multiples of 90 degrees.
    
    Parameters
    ----------
    data: numpy array
        your dataset, should be a stack of 2D images, i.e. a 3D numpy array
    size: int
        witdth and height of the patch
    numPix: int
        The number of pixels that is to be manipulated/masked N2V style.
    dataClean(optinal): numpy array 
        This dataset could hold your target data e.g. clean images.
        If it is not provided the function will use the image from 'data' N2V style
    counter (optinal): int
        the index of the next image to be used. 
        If not set, a random image will be used.
    augment: bool
        should the patches be randomy flipped and rotated?
    
    Returns
    ----------
    imgOut: numpy array 
        Cropped patch from training data
    imgOutC: numpy array
        Cropped target patch. If dataClean was provided it is used as source.
        Otherwise its generated N2V style from the training set
    mask: numpy array
        An image holding marking which pixels should be used to calculate gradients (value 1) and which not (value 0)
    counter: int
        The updated counter parameter, it is increased by one.
        When the counter reaches the end of the dataset, it is reset to zero and the dataset is shuffled.
    '''
    
    if counter is None:
        index=np.random.randint(0, data.shape[0])
    else:
        if counter>=data.shape[0]:
            counter=0
            np.random.shuffle(data)
        index=counter
        counter+=1

    if supervised:
        img=data[index,...,0]
        imgClean=data[index,...,1]
        manipulate=False
    else:
        img=data[index]
        imgClean=img
        manipulate=True
        
    imgOut, imgOutC, mask = randomCrop(img, size, numPix,
                                      imgClean=imgClean,
                                      augment=augment,
                                      manipulate = manipulate )
    
    return imgOut, imgOutC, mask, counter

def randomCrop(img, size, numPix, imgClean=None, augment=True, manipulate=True):
    '''
    Cuts out a random crop from an image.
    Manipulates pixels in the image (N2V style) and produces the corresponding mask of manipulated pixels.
    Patches are augmented by randomly deciding to mirror them and/or rotating them by multiples of 90 degrees.
    
    Parameters
    ----------
    img: numpy array
        your dataset, should be a 2D image
    size: int
        witdth and height of the patch
    numPix: int
        The number of pixels that is to be manipulated/masked N2V style.
    dataClean(optinal): numpy array 
        This dataset could hold your target data e.g. clean images.
        If it is not provided the function will use the image from 'data' N2V style
    augment: bool
        should the patches be randomy flipped and rotated?
        
    Returns
    ----------    
    imgOut: numpy array 
        Cropped patch from training data with pixels manipulated N2V style.
    imgOutC: numpy array
        Cropped target patch. Pixels have not been manipulated.
    mask: numpy array
        An image marking which pixels have been manipulated (value 1) and which not (value 0).
        In N2V or PN2V only these pixels should be used to calculate gradients.
    '''
    
    assert img.shape[0] >= size
    assert img.shape[1] >= size

    x = np.random.randint(0, img.shape[1] - size)
    y = np.random.randint(0, img.shape[0] - size)

    imgOut = img[y:y+size, x:x+size].copy()
    imgOutC= imgClean[y:y+size, x:x+size].copy()
    
    maxA=imgOut.shape[1]-1
    maxB=imgOut.shape[0]-1
    
    if manipulate:
        mask=np.zeros(imgOut.shape)
        hotPixels=getStratifiedCoords2D(numPix,imgOut.shape)
        for p in hotPixels:
            a,b=p[1],p[0]

            roiMinA=max(a-2,0)
            roiMaxA=min(a+3,maxA)
            roiMinB=max(b-2,0)
            roiMaxB=min(b+3,maxB)
            roi=imgOut[roiMinB:roiMaxB,roiMinA:roiMaxA]
            a_ = 2
            b_ = 2
            while a_==2 and b_==2:
                a_ = np.random.randint(0, roi.shape[1] )
                b_ = np.random.randint(0, roi.shape[0] )

            repl=roi[b_,a_]
            imgOut[b,a]=repl
            mask[b,a]=1.0
    else:
        mask=np.ones(imgOut.shape)

    if augment:
        rot=np.random.randint(0,4)
        imgOut=np.array(np.rot90(imgOut,rot))
        imgOutC=np.array(np.rot90(imgOutC,rot))
        mask=np.array(np.rot90(mask,rot))
        if np.random.choice((True,False)):
#             print (imgOut.shape)
            imgOut=np.array(np.flip(imgOut,1))
            imgOutC=np.array(np.flip(imgOutC,1))
            mask=np.array(np.flip(mask,1))


    return imgOut, imgOutC, mask


def trainingPred(my_train_data, net, dataCounter, size, bs, numPix, device, augment=True, supervised=True):
    '''
    This function will assemble a minibatch and process it using the a network.
    
    Parameters
    ----------
    my_train_data: numpy array
        Your training dataset, should be a stack of 2D images, i.e. a 3D numpy array
    net: a pytorch model
        the network we want to use
    dataCounter: int
        The index of the next image to be used. 
    size: int
        Witdth and height of the training patches that are to be used.
    bs: int 
        The batch size.
    numPix: int
        The number of pixels that is to be manipulated/masked N2V style.
    augment: bool
        should the patches be randomy flipped and rotated?
    Returns
    ----------
    samples: pytorch tensor
        The output of the network
    labels: pytorch tensor
        This is the tensor that was is used a target.
        It holds the raw unmanipulated patches.
    masks: pytorch tensor
        A tensor marking which pixels have been manipulated (value 1) and which not (value 0).
        In N2V or PN2V only these pixels should be used to calculate gradients.
    dataCounter: int
        The updated counter parameter, it is increased by one.
        When the counter reaches the end of the dataset, it is reset to zero and the dataset is shuffled.
    '''
    
    # Init Variables
    inputs= torch.zeros(bs,1,size,size)
    labels= torch.zeros(bs,size,size)
    masks= torch.zeros(bs,size,size)
   

    # Assemble mini batch
    for j in range(bs):
        im,l,m, dataCounter=randomCropFRI(my_train_data,
                                          size,
                                          numPix,
                                          counter=dataCounter,
                                          augment=augment,
                                          supervised=supervised)
        inputs[j,:,:,:]=utils.imgToTensor(im)
        labels[j,:,:]=utils.imgToTensor(l)
        masks[j,:,:]=utils.imgToTensor(m)

    # Move to GPU
    inputs_raw, labels, masks= inputs.to(device), labels.to(device), masks.to(device)

    # Move normalization parameter to GPU
    stdTorch=torch.Tensor(np.array(net.std)).to(device)
    meanTorch=torch.Tensor(np.array(net.mean)).to(device)
    
    # Forward step
    outputs = net((inputs_raw-meanTorch)/stdTorch) * 10.0 #We found that this factor can speed up training
    samples=(outputs).permute(1, 0, 2, 3)
    
    # Denormalize
    samples = samples * stdTorch + meanTorch
    
    return samples, labels, masks, dataCounter

def lossFunctionN2V(samples, labels, masks):
    '''
    The loss function as described in Eq. 7 of the paper.
    '''
        
    errors=(labels-torch.mean(samples,dim=0))**2

    # Average over pixels and batch
    loss= torch.sum( errors *masks  ) /torch.sum(masks)
    return loss

def lossFunctionPN2V(samples, labels, masks, noiseModel):
    '''
    The loss function as described in Eq. 7 of the paper.
    '''
    

    likelihoods=noiseModel.likelihood(labels,samples)
    likelihoods_avg=torch.log(torch.mean(likelihoods,dim=0,keepdim=True)[0,...] )

    # Average over pixels and batch
    loss= -torch.sum( likelihoods_avg *masks  ) /torch.sum(masks)
    return loss


def lossFunction(samples, labels, masks, noiseModel, pn2v, std=None):
    if pn2v:
        return lossFunctionPN2V(samples, labels, masks, noiseModel)
    else:
        return lossFunctionN2V(samples, labels, masks)/(std**2)



def trainNetwork(net, trainData, valData, te_Data_target, te_Data_source, noiseModel, postfix, device,
                 directory='.',
                 numOfEpochs=100, stepsPerEpoch=400,
                 batchSize=128, patchSize=64, learningRate=0.0004,
                 numMaskedPixels=64, 
                 virtualBatchSize=20, valSize=20,
                 augment=True,
                 supervised=False,
                 save_file_name = None,
                 ):
    '''
    Train a network using PN2V
    
    Parameters
    ----------
    net: 
        The network we want to train.
        The number of output channels determines the number of samples that are predicted.
    trainData: numpy array
        Our training data. A 3D array that is interpreted as a stack of 2D images.
    valData: numpy array
        Our validation data. A 3D array that is interpreted as a stack of 2D images.
    noiseModel: NoiseModel
        The noise model we will use during training.
    postfix: string
        This identifier is attached to the names of the files that will be saved during training.
    device: 
        The device we are using, e.g. a GPU or CPU
    directory: string
        The directory all files will be saved to.
    numOfEpochs: int
        Number of training epochs.
    stepsPerEpoch: int
        Number of gradient steps per epoch.
    batchSize: int
        The batch size, i.e. the number of patches processed simultainasly on the GPU.
    patchSize: int
        The width and height of the square training patches.
    learningRate: float
        The learning rate.
    numMaskedPixels: int
        The number of pixels that is to be manipulated/masked N2V style in every training patch.
    virtualBatchSize: int
        The number of batches that are processed before a gradient step is performed.
    valSize: int
        The number of validation patches processed after each epoch.
    augment: bool
        should the patches be randomy flipped and rotated? 
    
        
    Returns
    ----------    
    trainHist: numpy array 
        A numpy array containing the avg. training loss of each epoch.
    valHist: numpy array
        A numpy array containing the avg. validation loss after each epoch.
    '''
        
    # Calculate mean and std of data.
    # Everything that is processed by the net will be normalized and denormalized using these numbers.
    combined=np.concatenate((trainData))
    net.mean=np.mean(combined)
    net.std=np.std(combined)
    
    net.to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=learningRate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

    running_loss = 0.0
    stepCounter=0
    dataCounter=0

    result_psnr_arr = []
    result_ssim_arr = []
    result_time_arr = []
    result_denoised_img_arr = []
    result_te_loss_arr = []
    result_tr_loss_arr = []
        
    pn2v= (noiseModel is not None) and (not supervised)
    
    epoch = 0
    
    while stepCounter / stepsPerEpoch < numOfEpochs:  # loop over the dataset multiple times

        losses=[]
        optimizer.zero_grad()
        stepCounter+=1

        # Loop over our virtual batch
        for a in range (virtualBatchSize):
            outputs, labels, masks, dataCounter = trainingPred(trainData,
                                                               net,
                                                               dataCounter,
                                                               patchSize, 
                                                               batchSize,
                                                               numMaskedPixels,
                                                               device,
                                                               augment = augment,
                                                               supervised = supervised)
            loss=lossFunction(outputs, labels, masks, noiseModel, pn2v, net.std)
            loss.backward()
            running_loss += loss.item()
            losses.append(loss.item())

        optimizer.step()
        avgValLoss = []

        # We have reached the end of an epoch
        if stepCounter % stepsPerEpoch == stepsPerEpoch-1:
            running_loss=(np.mean(losses))
            losses=np.array(losses)
            utils.printNow("Epoch "+str(int(stepCounter / stepsPerEpoch))+" finished")
            utils.printNow("avg. loss: "+str(np.mean(losses))+"+-(2SEM)"+str(2.0*np.std(losses)/np.sqrt(losses.size)))
#             trainHist.append(np.mean(losses))
#             torch.save(net,os.path.join(directory,"last_"+postfix+".net"))

            valCounter=0
            net.train(False)
            losses=[]
            for i in range(valSize):
                outputs, labels, masks, valCounter = trainingPred(valData,
                                                                  net,
                                                                  valCounter,
                                                                  patchSize, 
                                                                  batchSize,
                                                                  numMaskedPixels,
                                                                  device,
                                                                  augment = augment,
                                                                  supervised = supervised)
                loss=lossFunction(outputs, labels, masks, noiseModel, pn2v, net.std)
                losses.append(loss.item())
                
            PSNR_arr=[]
            SSIM_arr=[]
            denoised_img_arr=[]
            time_arr = []
                
            for index in range(te_Data_target.shape[0]):    
                
                start = time.time()
                
                _, w, h = te_Data_target.shape
                remain = w%2
                
                im=te_Data_source[index,:w-remain,:h-remain]
                gt=te_Data_target[index,:w-remain,:h-remain] # The ground truth is the same for all images
                
                # We are using tiling to fit the image into memory
                # If you get an error try a smaller patch size (ps)
                n2vResult = prediction.tiledPredict(im, net ,ps=256, overlap=48,
                                                        device=device, noiseModel=None)
                
                inference_time = time.time()-start
                time_arr.append(inference_time)

#                 inputImgs.append(im)
                denoised_img_arr.append(n2vResult)

                rangePSNR=np.max(gt)-np.min(gt)
                PSNR_img=PSNR(gt, n2vResult,rangePSNR )
                PSNR_arr.append(PSNR_img)
                
                SSIM_img=get_SSIM(gt, n2vResult)
                SSIM_arr.append(SSIM_img)
                
                result_denoised_img_arr = denoised_img_arr.copy()
                
            mean_loss = np.mean(running_loss)
            mean_psnr = np.mean(PSNR_arr)
            mean_ssim = np.mean(SSIM_arr)
            mean_time = np.mean(time_arr)
            
            result_psnr_arr.append(mean_psnr)
            result_ssim_arr.append(mean_ssim)
            result_time_arr.append(mean_time)
#             result_te_loss_arr.append(mean_te_loss)
            result_tr_loss_arr.append(mean_loss)
                
                
            net.train(True)
            avgValLoss=np.mean(losses)
#             if len(valHist)==0 or avgValLoss < np.min(np.array(valHist)):
#                 torch.save(net,os.path.join(directory,"best_"+postfix+".net"))
#             valHist.append(avgValLoss)
            scheduler.step(avgValLoss)
            epoch= (stepCounter / stepsPerEpoch)
#             np.save(os.path.join(directory,"history"+postfix+".npy"), (np.array( [np.arange(epoch),trainHist,valHist ] ) ) )

            print ('Tr loss : ', round(running_loss,4), ' PSNR : ', round(mean_psnr,2), ' SSIM : ', round(mean_ssim,4),' Best Time : ', round(mean_time,2)) 

            sio.savemat('../../result_data/'+ save_file_name + '_result',{'tr_loss_arr':result_tr_loss_arr,'psnr_arr':result_psnr_arr, 'ssim_arr':result_ssim_arr,'time_arr':result_time_arr, 'denoised_img':result_denoised_img_arr[:10]})
            torch.save(net.state_dict(), '../../weights/'+save_file_name + '.w')

    utils.printNow('Finished Training')
    return 