import torch
import torch.nn as nn
import torch.nn.functional as F

from core.layers import Residual_module, New1_layer, New2_layer, New3_layer, Receptive_attention

class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

class New_model(nn.Module):
    def __init__(self, channel = 1, output_channel = 1, filters = 64, num_of_layers=10, mul = 1, case = 'FBI_Net', output_type='linear', sigmoid_value = 0.1):
        super(New_model, self).__init__()
        
        self.case = case

        self.new1 = New1_layer(channel, filters, mul = mul, case = case).cuda()
        self.new2 = New2_layer(filters, filters, mul = mul, case = case).cuda()
        
        self.num_layers = num_of_layers
        self.output_type = output_type
        self.sigmoid_value = sigmoid_value

        dilated_value = 3
        
        for layer in range (num_of_layers-2):
            self.add_module('new_' + str(layer), New3_layer(filters, filters, dilated_value, mul = mul, case = case).cuda())
            
        self.residual_module = Residual_module(filters, mul)
        self.activation = nn.PReLU(filters,0).cuda()
        self.output_layer = nn.Conv2d(in_channels=filters, out_channels=output_channel, kernel_size = 1).cuda()
        
        if self.output_type == 'sigmoid':
            self.sigmoid=nn.Sigmoid().cuda()
        
        self.new = AttrProxy(self, 'new_')

    def forward(self, x):
        
        if self.case == 'FBI_Net' or self.case == 'case2' or self.case == 'case3' or self.case == 'case4':

            output, output_new = self.new1(x)
            output_sum = output
            output, output_new = self.new2(output, output_new)
            output_sum = output + output_sum

            for i, (new_layer)  in enumerate(self.new):

                output, output_new  = new_layer(output, output_new)
                output_sum = output + output_sum

                if i == self.num_layers - 3:
                    break

            final_output = self.activation(output_sum/self.num_layers)
            final_output = self.residual_module(final_output)
            final_output = self.output_layer(final_output)
            
        else:

            output, output_new = self.new1(x)
            output, output_new = self.new2(output, output_new)

            for i, (new_layer)  in enumerate(self.new):

                output, output_new  = new_layer(output, output_new)

                if i == self.num_layers - 3:
                    break

            final_output = self.activation(output)
            final_output = self.residual_module(final_output)
            final_output = self.output_layer(final_output)
            
        if self.output_type=='sigmoid':
               final_output[:,0]=(torch.ones_like(final_output[:,0])*self.sigmoid_value)*self.sigmoid(final_output[:,0])

        return final_output
    
