import torch
import torch.nn as nn
import torch.nn.functional as F

from core.layers import Residual_module, New1_layer, New2_layer, New3_layer

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
    