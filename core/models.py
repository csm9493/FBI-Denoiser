import torch
import torch.nn as nn
import torch.nn.functional as F

from core.layers import Residual_module, New1_layer, New2_layer, New3_layer
from core.layers import QED_first_layer, QED_layer, Average_layer

class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

class New_model(nn.Module):
    def __init__(self, channel = 1, output_channel = 1, filters = 64, num_of_layers=10):
        super(New_model, self).__init__()

        self.new1 = New1_layer(channel, filters).cuda()
        self.new2 = New2_layer(filters, filters).cuda()
        
        self.num_layers = num_of_layers

        dilated_value = 3
        
        for layer in range (num_of_layers-2):
            self.add_module('new_' + str(layer), New3_layer(filters, filters, dilated_value).cuda())
            
        self.activation = nn.PReLU(filters,0).cuda()
        self.output_layer = nn.Conv2d(in_channels=filters, out_channels=output_channel, kernel_size = 1).cuda()
        
        self.new = AttrProxy(self, 'new_')
        self.residual_module = AttrProxy(self, 'residual_module_')

    def forward(self, x):
        
        residual_output_arr = []
        
        x, output_new = self.new1(x)
        x, output_new = self.new2(x, output_new)
        
        for i, (new_layer)  in enumerate(self.new):

            x, output_new  = new_layer(x, output_new)
            if i >= self.num_layers - 3:
                break
                
        output = self.activation(x)
        output = self.output_layer(x)
        
        return output
    
class New_model_ablation(nn.Module):
    def __init__(self, channel = 1, output_channel = 1, filters = 64, num_of_layers=8, case = 'case1'):
        super(New_model_ablation, self).__init__()

        self.new1 = New1_layer(channel, filters, case).cuda()
        self.new2 = New2_layer(filters, filters, case).cuda()
        
        self.num_layers = num_of_layers

        dilated_value = 3
        
        for layer in range (num_of_layers-2):
            self.add_module('new_' + str(layer), New3_layer(filters, filters, dilated_value, case).cuda())
            
        self.activation = nn.PReLU(filters,0).cuda()
        self.output_layer = nn.Conv2d(in_channels=filters, out_channels=output_channel, kernel_size = 1).cuda()
        
        self.new = AttrProxy(self, 'new_')
        self.residual_module = AttrProxy(self, 'residual_module_')

    def forward(self, x):
        
        residual_output_arr = []
        
        x = self.new1(x)
        x = self.new2(x, None)
        
        for i, (new_layer)  in enumerate(self.new):

            x  = new_layer(x, None)
            if i >= self.num_layers - 3:
                break

        output = self.activation(x)
        output = self.output_layer(output)
        
        return output
    

class FC_AIDE(nn.Module):
    def __init__(self, channel = 1, output_channel = 2, filters = 64, num_of_layers=10):
        super(FC_AIDE, self).__init__()
        
        self.qed_first_layer = QED_first_layer(channel, filters).cuda()
        self.avg_first_layer = Average_layer(filters)
        self.residual_module_first_layer = Residual_module(filters)
        
        self.num_layers = num_of_layers

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
        
        return output