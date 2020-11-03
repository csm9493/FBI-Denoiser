import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from functools import partial

def init_weights(net, init_type='kaiming', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def weights_init_kaiming(lyr):
	r"""Initializes weights of the model according to the "He" initialization
	method described in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015), using a
    normal distribution.
	This function is to be called by the torch.nn.Module.apply() method,
	which applies weights_init_kaiming() to every layer of the model.
	"""
	classname = lyr.__class__.__name__
	if classname.find('Conv') != -1:
		lyr.weight.data = nn.init.kaiming_normal_(lyr.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		nn.init.kaiming_normal_(lyr.weight.data, a=0, mode='fan_in')
	elif classname.find('BatchNorm') != -1:
		lyr.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).\
			clamp_(-0.025, 0.025)
		nn.init.constant_(lyr.bias.data, 0.0)

class TrimmedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        if 'dilation' in kwargs:
            self.dilation = kwargs['dilation']
            kwargs.pop('dilation')
        else:
            self.dilation = 1

        if 'direction' in kwargs:
            self.direction = kwargs['direction']
            kwargs.pop('direction')
        else:
            self.direction = 0

        super(TrimmedConv2d, self).__init__(*args, **kwargs)

        self.slide_winsize = self.weight.shape[2]*self.weight.shape[3]
        self.last_size = torch.zeros(2)
        self.feature_mask=None
        self.mask_ratio=None
        self.weight_mask=None
        self.mask_ratio_dict=dict()
        self.feature_mask_dict=dict()


    def update_mask(self):
        with torch.no_grad():
            self.feature_mask=self.feature_mask_dict[str(self.direction)].to(self.weight.device)
            self.mask_ratio=self.mask_ratio_dict[str(self.direction)].to(self.weight.device)
            self.weight_mask=self.get_weight_mask().to(self.weight.device)

    def get_weight_mask(self,direction=None):
        weight = np.ones((1, 1, self.kernel_size[0], self.kernel_size[1]))
        weight[:, :, self.kernel_size[0] // 2, self.kernel_size[1] // 2] = 0
        return torch.tensor(weight.copy(),dtype=torch.float32)

    def update_feature_mask_dict(self,input_h,input_w):
        with torch.no_grad():
            for direct in range(0,1): 
                mask = torch.ones(1, 1, int(input_h), int(input_w))
                weight_mask=self.get_weight_mask(direct)
                (pad_h,pad_w)=self.padding
                pad=torch.nn.ZeroPad2d((pad_w,pad_w,pad_h,pad_h))
                feature_mask = F.conv2d(pad(mask), weight_mask, bias=None, stride=self.stride, dilation=self.dilation, groups=1)
                mask_ratio = self.slide_winsize / (feature_mask + 1e-8)
                # mask_ratio=torch.sqrt(mask_ratio)
                feature_mask = torch.clamp(feature_mask, 0, 1)
                mask_ratio = torch.mul(mask_ratio, feature_mask)
                self.mask_ratio_dict[str(direct)]=mask_ratio
                self.feature_mask_dict[str(direct)]=feature_mask

    def updata_last_size(self,h,w):
        self.last_size.copy_(torch.tensor((h,w),dtype=torch.int32))

    def forward(self, input):
        if (int(self.last_size[0].item()),int(self.last_size[1].item()))!= (int(input.data.shape[2]), int(input.data.shape[3])):
            self.update_feature_mask_dict(input.data.shape[2],input.data.shape[3])
            self.update_mask()
            self.updata_last_size(input.data.shape[2],input.data.shape[3])
        if self.feature_mask is None or self.mask_ratio is None or self.weight_mask is None:
            #self.update_feature_mask_dict()
            self.update_mask()
        #if self.feature_mask.device  != self.weight.device or self.mask_ratio.device != self.weight.device or self.weight_mask.device!=self.weight.device:
        #    with torch.no_grad():
        w=torch.mul(self.weight, self.weight_mask)
        raw_out = F.conv2d(input,w,self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.feature_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)
        return output


class MaskConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        if 'dilation' in kwargs:
            self.dilation = kwargs['dilation']
            kwargs.pop('dilation')
        else:
            self.dilation = 1

        if 'direction' in kwargs:
            self.direction = kwargs['direction']
            kwargs.pop('direction')
        else:
            self.direction = 0
            
        super(MaskConv2d, self).__init__(*args, **kwargs)
        self.weight_mask = self.get_weight_mask()


    # remove the center position, [1 1 1;1 0 1;1 1 1]
    def get_weight_mask(self):
        weight = np.ones((1, 1, self.kernel_size[0], self.kernel_size[1]))
        weight[:, :, self.kernel_size[0] // 2, self.kernel_size[1] // 2] = 0
        return torch.tensor(weight.copy(), dtype=torch.float32)

    def forward(self, input):
        if self.weight_mask.type() != self.weight.type():
            with torch.no_grad():
                self.weight_mask = self.weight_mask.type(self.weight.type())
        w=torch.mul(self.weight,self.weight_mask)
        output = F.conv2d(input, w, self.bias, self.stride,
                           self.padding, self.dilation, self.groups)
        return output

def BlindSpotConv(in_planes, out_planes, kernel_size, stride=1, dilation=1, bias=False, conv_type='Trimmed'):
    if conv_type.lower()=='trimmed':
        return TrimmedConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
            padding=(kernel_size+(kernel_size-1)*(dilation-1)-1)//2, dilation=dilation, bias=bias, direction=0)
    elif conv_type.lower()=='mask':
        return MaskConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
            padding=(kernel_size+(kernel_size-1)*(dilation-1)-1)//2, dilation=dilation, bias=bias, direction=0)
    else:
        raise BaseException("Invalid Conv Type!")

class Inception_block(nn.Module):
    def __init__(self, inplanes, kernel_size, dilation, bias, activate_fun):
        super(Inception_block, self).__init__()
        #
        if activate_fun == 'Relu':
            # self.relu = nn.ReLU(inplace=True)
            self.relu = partial(nn.ReLU, inplace=True)
        elif activate_fun == 'LeakyRelu':
            # self.relu = nn.LeakyReLU(0.1)
            self.relu = partial(nn.LeakyReLU, negative_slope=0.1)
        else:
            raise ValueError('activate_fun [%s] is not found.' % (activate_fun))
        #
        pad_size = (kernel_size+(kernel_size-1)*(dilation-1)-1)//2
        # inception_br1 ----------------------------------------------
        lyr_br1=[]
        # 1x1 conv
        lyr_br1.append(nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=bias))
        lyr_br1.append(self.relu())
        # # case1: two 3x3 dilated-conv
        # lyr_br1.append(nn.Conv2d(inplanes, inplanes, kernel_size, padding=pad_size, dilation=dilation, bias=bias))
        # lyr_br1.append(self.relu())
        # lyr_br1.append(nn.Conv2d(inplanes, inplanes, kernel_size, padding=pad_size, dilation=dilation, bias=bias))
        # lyr_br1.append(self.relu())
        # case2: one 5x5 dilated-conv
        tmp_kernel_size = 5
        tmp_pad_size = (tmp_kernel_size+(tmp_kernel_size-1)*(dilation-1)-1)//2
        lyr_br1.append(nn.Conv2d(inplanes, inplanes, kernel_size=tmp_kernel_size, padding=tmp_pad_size, dilation=dilation, bias=bias))
        lyr_br1.append(self.relu())
        self.inception_br1=nn.Sequential(*lyr_br1)
        init_weights(self.inception_br1)
        #
        # inception_br2 ----------------------------------------------
        lyr_br2=[]
        # 1x1 conv
        lyr_br2.append(nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=bias))
        lyr_br2.append(self.relu())
        # 3x3 dilated-conv
        lyr_br2.append(nn.Conv2d(inplanes, inplanes, kernel_size, padding=pad_size, dilation=dilation, bias=bias))
        lyr_br2.append(self.relu())
        self.inception_br2=nn.Sequential(*lyr_br2)
        init_weights(self.inception_br2)
        #
        # inception_br3 ----------------------------------------------
        lyr_br3=[]
        # 1x1 conv
        lyr_br3.append(nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=bias))
        lyr_br3.append(self.relu())
        self.inception_br3=nn.Sequential(*lyr_br3)
        init_weights(self.inception_br3)
        # Concat three inception branches
        self.concat = nn.Conv2d(inplanes*3,inplanes,kernel_size=1,bias=bias)
        self.concat.apply(weights_init_kaiming)
        # 1x1 convs
        lyr=[]
        lyr.append(nn.Conv2d(inplanes,inplanes,kernel_size=1,bias=bias))
        lyr.append(self.relu())
        lyr.append(nn.Conv2d(inplanes,inplanes,kernel_size=1,bias=bias))
        lyr.append(self.relu())
        self.middle_1x1_convs=nn.Sequential(*lyr)
        init_weights(self.middle_1x1_convs)
  

    def forward(self, x):
        residual = x
        x1 = self.inception_br1(x)
        x2 = self.inception_br2(x)
        x3 = self.inception_br3(x)
        out = torch.cat((x1, x2, x3), dim=1)
        out = self.concat(out)
        out = torch.relu_(out)
        out = out + residual
        out = self.middle_1x1_convs(out)
        return out


class DBSN_branch(nn.Module):
    def __init__(self, inplanes, bs_conv_type, bs_conv_bias, bs_conv_ks, block_num, activate_fun):
        super(DBSN_branch, self).__init__()
        # 
        if activate_fun == 'Relu':
            # self.relu = nn.ReLU(inplace=True)
            self.relu = partial(nn.ReLU, inplace=True)
        elif activate_fun == 'LeakyRelu':
            # self.relu = nn.LeakyReLU(0.1)
            self.relu = partial(nn.LeakyReLU, negative_slope=0.1)
        else:
            raise ValueError('activate_fun [%s] is not found.' % (activate_fun))
        #
        dilation_base=(bs_conv_ks+1)//2
        #
        lyr=[]
        lyr.append(BlindSpotConv(inplanes, inplanes, bs_conv_ks, stride=1, dilation=1, bias=bs_conv_bias, conv_type=bs_conv_type))
        lyr.append(self.relu())
        lyr.append(nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=bs_conv_bias))
        lyr.append(self.relu())
        lyr.append(nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=bs_conv_bias))
        lyr.append(self.relu())
        #
        for i in range(block_num):
            lyr.append(Inception_block(inplanes, kernel_size=3, dilation=dilation_base, bias=bs_conv_bias, activate_fun=activate_fun))
        #
        lyr.append(nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=bs_conv_bias))
        self.branch=nn.Sequential(*lyr)
        init_weights(self.branch)

    def forward(self,x):
        return self.branch(x)

class DBSN_Model(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch, 
                blindspot_conv_type, blindspot_conv_bias,
                br1_blindspot_conv_ks, br1_block_num, 
                br2_blindspot_conv_ks, br2_block_num,
                activate_fun):
        super(DBSN_Model,self).__init__()
        #
        if activate_fun == 'Relu':
            # self.relu = nn.ReLU(inplace=True)
            self.relu = partial(nn.ReLU, inplace=True)
        elif activate_fun == 'LeakyRelu':
            # self.relu = nn.LeakyReLU(0.1)
            self.relu = partial(nn.LeakyReLU, negative_slope=0.1)
        else:
            raise ValueError('activate_fun [%s] is not found.' % (activate_fun))
        # Head of DBSN
        lyr = []
        lyr.append(nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=blindspot_conv_bias))
        lyr.append(self.relu())
        self.dbsn_head = nn.Sequential(*lyr)
        init_weights(self.dbsn_head)

        self.br1 = DBSN_branch(mid_ch, blindspot_conv_type, blindspot_conv_bias, br1_blindspot_conv_ks, br1_block_num, activate_fun)
        self.br2 = DBSN_branch(mid_ch, blindspot_conv_type, blindspot_conv_bias, br2_blindspot_conv_ks, br2_block_num, activate_fun)

        # Concat two branches
        self.concat = nn.Conv2d(mid_ch*2,mid_ch,kernel_size=1,bias=blindspot_conv_bias)
        self.concat.apply(weights_init_kaiming)
        # 1x1 convs
        lyr=[]
        lyr.append(nn.Conv2d(mid_ch,mid_ch,kernel_size=1,bias=blindspot_conv_bias))
        lyr.append(self.relu())
        lyr.append(nn.Conv2d(mid_ch,mid_ch,kernel_size=1,bias=blindspot_conv_bias))
        lyr.append(self.relu())
        lyr.append(nn.Conv2d(mid_ch,out_ch,kernel_size=1,bias=blindspot_conv_bias))
        self.dbsn_tail=nn.Sequential(*lyr)
        init_weights(self.dbsn_tail)

    def forward(self, x):
        x = self.dbsn_head(x)
        x1 = self.br1(x)     
        x2 = self.br2(x)
        x_concat = torch.cat((x1,x2), dim=1)
        x = self.concat(x_concat)
        return self.dbsn_tail(x), x