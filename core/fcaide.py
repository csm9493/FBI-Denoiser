import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import QED_first_layer, QED_layer

class Average_layer(nn.Module):
    def __init__(self, in_ch):
        super(Average_layer, self).__init__()
        self.prelu = nn.PReLU(in_ch,0).cuda()

    def forward(self, inputs):

        mean = torch.mean(torch.stack(inputs), dim=0)
        output = self.prelu(mean)
        
        return output

class Residual_module(nn.Module):
    def __init__(self, in_ch):
        super(Residual_module, self).__init__()

        self.activation1 = nn.PReLU(in_ch,0).cuda()
        self.activation2 = nn.PReLU(in_ch,0).cuda()
        
        self.conv1_1by1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size = 1)
        self.conv2_1by1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size = 1)

    def forward(self, input):
        
        output_residual = self.conv1_1by1(input)
        output_residual = self.activation1(output_residual)
        output_residual = self.conv2_1by1(output_residual)
        
        output = torch.mean(torch.stack([input, output_residual]), dim=0)
        output = self.activation2(output)
        
        return output

class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

class FC_AIDE(nn.Module):
    def __init__(self, channel = 1, output_channel = 2, filters = 64, num_of_layers=10, output_type='linear', sigmoid_value = 0.1):
        super(FC_AIDE, self).__init__()
        
        self.qed_first_layer = QED_first_layer(channel, filters).cuda()
        self.avg_first_layer = Average_layer(filters)
        self.residual_module_first_layer = Residual_module(filters)
        
        self.num_layers = num_of_layers
        self.output_type = output_type
        self.sigmoid_value = sigmoid_value

        dilated_value = 1
        
        for layer in range (num_of_layers-1):
            self.add_module('qed_' + str(layer), QED_layer(filters, filters, dilated_value).cuda())
            self.add_module('avg_' + str(layer), Average_layer(filters))
            self.add_module('residual_module_' + str(layer), Residual_module(filters).cuda())
            dilated_value += 1
            
        self.output_avg_layer = Average_layer(filters)
        self.output_conv1 =  nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size = 1).cuda()
        self.output_prelu1 = nn.PReLU(filters,0).cuda()
        self.output_residual_module = Residual_module(filters).cuda()
        
        self.output_layer = nn.Conv2d(in_channels=filters, out_channels=output_channel, kernel_size = 1).cuda()
        
        if self.output_type == 'sigmoid':
            self.sigmoid=nn.Sigmoid().cuda()
        
        self.qed = AttrProxy(self, 'qed_')
        self.avg = AttrProxy(self, 'avg_')
        self.residual_module = AttrProxy(self, 'residual_module_')
    
    def forward(self, x):
        
        residual_output_arr = []
        
        qed_output = self.qed_first_layer(x)
        avg_output = self.avg_first_layer(qed_output)
        residual_output = self.residual_module_first_layer(avg_output)
        
        residual_output_arr.append(residual_output)

        for i, (qed_layer, avg_layer, residual_layer)  in enumerate(zip(self.qed, self.avg, self.residual_module)):

            qed_output = qed_layer(qed_output)
            avg_output = avg_layer(qed_output)
            residual_output = residual_layer(avg_output)
            residual_output_arr.append(residual_output)
            
            if i >= self.num_layers - 2:
                break
            
        output = self.output_avg_layer(residual_output_arr)
        output = self.output_conv1(output)
        output = self.output_prelu1(output)
        output = self.output_residual_module(output)
        output = self.output_layer(output)
        
        if self.output_type=='sigmoid':
               output[:,0]=(torch.ones_like(output[:,0])*self.sigmoid_value)*self.sigmoid(output[:,0])
        
        return output