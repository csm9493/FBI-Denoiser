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
    def __init__(self, channel = 1, output_channel = 1, filters = 64, num_of_layers=10, mul = 1, case = 'final'):
        super(New_model, self).__init__()
        
        self.case = case

        self.new1 = New1_layer(channel, filters, mul = mul, case = case).cuda()
        self.new2 = New2_layer(filters, filters, mul = mul, case = case).cuda()
        
        self.num_layers = num_of_layers

        dilated_value = 3
        
        for layer in range (num_of_layers-2):
            self.add_module('new_' + str(layer), New3_layer(filters, filters, dilated_value, mul = mul, case = case).cuda())
            
        self.residual_module = Residual_module(filters, mul)
        self.activation = nn.PReLU(filters,0).cuda()
        self.output_layer = nn.Conv2d(in_channels=filters, out_channels=output_channel, kernel_size = 1).cuda()
        
        self.new = AttrProxy(self, 'new_')

    def forward(self, x):
        
        if self.case == 'final' or self.case == 'case2' or self.case == 'case3' or self.case == 'case4':

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

        return final_output
    

# class New_model_attention(nn.Module):
#     def __init__(self, channel = 1, output_channel = 1, filters = 64, num_of_layers=10):
#         super(New_model_attention, self).__init__()

#         self.new1 = New1_layer(channel, filters).cuda()
#         self.new2 = New2_layer(filters, filters).cuda()
        
#         self.num_layers = num_of_layers

#         dilated_value = 3
        
#         for layer in range (num_of_layers-2):
#             self.add_module('new_' + str(layer), New3_layer(filters, filters, dilated_value).cuda())
            
#         self.attention = Receptive_attention(filters)
            
#         self.residual_module = Residual_module(filters)
#         self.activation = nn.PReLU(filters,0).cuda()
#         self.output_layer = nn.Conv2d(in_channels=filters, out_channels=output_channel, kernel_size = 1).cuda()
        
#         self.new = AttrProxy(self, 'new_')

#     def forward(self, x):
        
#         residual_output_arr = []
        
#         output, output_new = self.new1(x)
#         output_sum = output
#         output, output_new = self.new2(output, output_new)
# #         output_sum = output + output_sum
#         output_sum = torch.stack([output_sum, output], axis = 0)
        
#         for i, (new_layer)  in enumerate(self.new):

#             output, output_new  = new_layer(output, output_new)
#             output_sum = torch.cat([output_sum, output.reshape((1,output.shape[0],output.shape[1],output.shape[2],output.shape[3]))], axis = 0)
            
#             if i == self.num_layers - 3:
#                 break
                
#         output = self.attention(output, output_sum)
                
#         final_output = self.activation(output)
#         final_output = self.residual_module(final_output)
#         final_output = self.output_layer(final_output)
        
#         return final_output
    
# class New_model_ablation_case1(nn.Module):
#     def __init__(self, channel = 1, output_channel = 1, filters = 64, num_of_layers=10, case = 'case1'):
#         super(New_model_ablation_case3, self).__init__()

#         self.new1 = New1_layer(channel, filters, case).cuda()
#         self.new2 = New2_layer(filters, filters, case).cuda()
        
#         self.num_layers = num_of_layers

#         dilated_value = 3
        
#         for layer in range (num_of_layers-2):
#             self.add_module('new_' + str(layer), New3_layer(filters, filters, dilated_value, case).cuda())
            
#         self.residual_module = Residual_module(filters)
#         self.activation = nn.PReLU(filters,0).cuda()
#         self.output_layer = nn.Conv2d(in_channels=filters, out_channels=output_channel, kernel_size = 1).cuda()
        
#         self.new = AttrProxy(self, 'new_')

#     def forward(self, x):
        
#         residual_output_arr = []
        
#         output, output_new = self.new1(x)
#         output, output_new = self.new2(output, output_new)
        
#         for i, (new_layer)  in enumerate(self.new):

#             output, output_new  = new_layer(output, output_new)
            
#             if i == self.num_layers - 3:
#                 break
                
#         final_output = self.activation(output)
#         final_output = self.residual_module(final_output)
#         final_output = self.output_layer(final_output)
        
#         return final_output
    
# class New_model_ablation(nn.Module):
#     def __init__(self, channel = 1, output_channel = 1, filters = 64, num_of_layers=8, case = 'case1'):
#         super(New_model_ablation, self).__init__()

#         self.new1 = New1_layer(channel, filters, case).cuda()
#         self.new2 = New2_layer(filters, filters, case).cuda()
        
#         self.num_layers = num_of_layers

#         dilated_value = 3
        
#         for layer in range (num_of_layers-2):
#             self.add_module('new_' + str(layer), New3_layer(filters, filters, dilated_value, case).cuda())
            
#         self.residual_module = Residual_module(filters)
#         self.activation = nn.PReLU(filters,0).cuda()
#         self.output_layer = nn.Conv2d(in_channels=filters, out_channels=output_channel, kernel_size = 1).cuda()
        
#         self.new = AttrProxy(self, 'new_')

#     def forward(self, x):
        
#         residual_output_arr = []
        
#         output, output_new = self.new1(x)
#         output_sum = output
#         output, output_new = self.new2(output, output_new)
#         output_sum = output + output_sum
        
#         for i, (new_layer)  in enumerate(self.new):

#             output, output_new  = new_layer(output, output_new)
#             output_sum = output + output_sum
            
#             if i == self.num_layers - 3:
#                 break
                
#         final_output = self.activation(output_sum/self.num_layers)
#         final_output = self.residual_module(final_output)
#         final_output = self.output_layer(final_output)
        
        
#         return final_output
    