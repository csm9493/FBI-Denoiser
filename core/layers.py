import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class New1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(New1, self).__init__()
       
        self.mask = torch.from_numpy(np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.float32)).cuda()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, padding = 1, kernel_size = 3)

    def forward(self, x):
        self.conv1.weight.data =  self.conv1.weight * self.mask
        x = self.conv1(x)
        
        return x   
    
class New2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(New2, self).__init__()
        
        self.mask = torch.from_numpy(np.array([[0,1,0,1,0],[1,0,0,0,1],[0,0,1,0,0],[1,0,0,0,1],[0,1,0,1,0]], dtype=np.float32)).cuda()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, padding = 2, kernel_size = 5)

    def forward(self, x):
        self.conv1.weight.data =  self.conv1.weight * self.mask
        x = self.conv1(x)

        return x
    
class New3(nn.Module):
    def __init__(self, in_ch, out_ch, dilated_value):
        super(New3, self).__init__()
        
        self.mask = torch.from_numpy(np.array([[1,0,1],[0,1,0],[1,0,1]], dtype=np.float32)).cuda()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size = 3, padding=dilated_value, dilation=dilated_value)

    def forward(self, x):
        self.conv1.weight.data =  self.conv1.weight * self.mask
        x = self.conv1(x)

        return x
    
class Residual_module(nn.Module):
    def __init__(self, in_ch, mul = 1):
        super(Residual_module, self).__init__()
        
        self.activation1 = nn.PReLU(in_ch*mul,0).cuda()
        self.activation2 = nn.PReLU(in_ch,0).cuda()
            
        self.conv1_1by1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch*mul, kernel_size = 1)
        self.conv2_1by1 = nn.Conv2d(in_channels=in_ch*mul, out_channels=in_ch, kernel_size = 1)

    def forward(self, input):

        output_residual = self.conv1_1by1(input)
        output_residual = self.activation1(output_residual)
        output_residual = self.conv2_1by1(output_residual)
        
        output = (input + output_residual) / 2.
        output = self.activation2(output)
        
        return output
    
class Gaussian(nn.Module):
    def forward(self,input):
        return torch.exp(-torch.mul(input,input))
    

class Receptive_attention(nn.Module):
    def __init__(self, in_ch, at_type = 'softmax'):
        super(Receptive_attention, self).__init__()
        
        self.activation1 = nn.ReLU().cuda()
        self.activation2 = nn.ReLU().cuda()
        self.activation3 = nn.PReLU(in_ch,0).cuda()
            
        self.conv1_1by1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch*4, kernel_size = 1)
        self.conv2_1by1 = nn.Conv2d(in_channels=in_ch*4, out_channels=in_ch*4, kernel_size = 1)
        self.conv3_1by1 = nn.Conv2d(in_channels=in_ch*4, out_channels=9, kernel_size = 1)
        self.at_type = at_type
        if at_type == 'softmax':
            self.softmax = nn.Softmax()
        else:
            self.gaussian = Gaussian()
            self.sigmoid = nn.Sigmoid()
            

    def forward(self, input, receptive):

        if self.at_type == 'softmax':
            output_residual = self.conv1_1by1(input)
            output_residual = self.activation1(output_residual)
            output_residual = self.conv2_1by1(output_residual)
            output_residual = self.activation2(output_residual)
            output_residual = self.conv3_1by1(output_residual)
            output_residual = F.adaptive_avg_pool2d(output_residual, (1, 1))
    #         output_residual = self.Gaussian(output_residual)
            output_residual = self.softmax(output_residual).permute((1,0,2,3)).unsqueeze(-1)
        else:
            
            output_residual = self.conv1_1by1(input)
            output_residual = self.activation1(output_residual)
            output_residual = self.conv2_1by1(output_residual)
            output_residual = self.activation2(output_residual)
            output_residual = self.conv3_1by1(output_residual)
            output_residual = F.adaptive_avg_pool2d(output_residual, (1, 1))
            output_residual = self.gaussian(output_residual)
            output_residual = self.sigmoid(output_residual).permute((1,0,2,3)).unsqueeze(-1)
        
        output = torch.sum(receptive * output_residual, dim = 0)
        output = self.activation3(output)
        
        return output
    
class New1_layer(nn.Module):
    def __init__(self, in_ch, out_ch, case = 'final', mul = 1):
        super(New1_layer, self).__init__()
        self.case = case
        self.new1 = New1(in_ch,out_ch).cuda()
        if case == 'case1' or case == 'case2' or case == 'case7' or case == 'final':
            self.residual_module = Residual_module(out_ch, mul)
            
        self.activation_new1 = nn.PReLU(in_ch,0).cuda()
        

    def forward(self, x):
        
        
        if self.case == 'case1' or self.case =='case2'  or self.case =='case7' or self.case == 'final': # plain NN architecture wo residual module and residual connection
            
            output_new1 = self.new1(x)
            output_new1 = self.activation_new1(output_new1)
            output = self.residual_module(output_new1)

            return output, output_new1

        else: # final model
        
            output_new1 = self.new1(x)
            output = self.activation_new1(output_new1)

            return output, output_new1
   
class New2_layer(nn.Module):
    def __init__(self, in_ch, out_ch, case = 'final', mul = 1):
        super(New2_layer, self).__init__()
        
        self.case = case
        
        self.new2 = New2(in_ch,out_ch).cuda()
        self.activation_new1 = nn.PReLU(in_ch,0).cuda()
        if case == 'case1' or case == 'case2' or case == 'case7' or case == 'final':
            self.residual_module = Residual_module(out_ch, mul)
        if case == 'case1' or case == 'case3' or case == 'case6' or case == 'final':
            self.activation_new2 = nn.PReLU(in_ch,0).cuda()
        

    def forward(self, x, output_new):
        
        if self.case == 'case1': #
            
            output_new2 = self.new2(output_new)
            output_new2 = self.activation_new1(output_new2)

            output = (output_new2 + x) / 2.
            output = self.activation_new2(output)
            output = self.residual_module(output)

            return output, output_new2
            

        elif self.case == 'case2' or self.case == 'case7': #
            
            output_new2 = self.new2(x)
            output_new2 = self.activation_new1(output_new2)

            output = output_new2
            output = self.residual_module(output)

            return output, output_new2
        
        elif self.case == 'case3' or self.case == 'case6': #
            
            output_new2 = self.new2(output_new)
            output_new2 = self.activation_new1(output_new2)

            output = (output_new2 + x) / 2.
            output = self.activation_new2(output)

            return output, output_new2

        elif self.case == 'case4': #
            
            output_new2 = self.new2(x)
            output_new2 = self.activation_new1(output_new2)

            output = output_new2
            
            return output, output_new2
        
        elif self.case == 'case5' : #
            
            output_new2 = self.new2(x)
            output_new2 = self.activation_new1(output_new2)

            output = output_new2
            
            return output, output_new2
        
        else:

            output_new2 = self.new2(output_new)
            output_new2 = self.activation_new1(output_new2)

            output = (output_new2 + x) / 2.
            output = self.activation_new2(output)
            output = self.residual_module(output)

            return output, output_new2
            
    
class New3_layer(nn.Module):
    def __init__(self, in_ch, out_ch, dilated_value=3, case = 'final', mul = 1):
        super(New3_layer, self).__init__()
        
        self.case = case
        
        self.new3 = New3(in_ch,out_ch,dilated_value).cuda()
        self.activation_new1 = nn.PReLU(in_ch,0).cuda()
        if case == 'case1' or case == 'case2'  or case == 'case7' or case == 'final':
            self.residual_module = Residual_module(out_ch, mul)
        if case == 'case1' or case == 'case3' or case == 'case6'or case == 'final':
            self.activation_new2 = nn.PReLU(in_ch,0).cuda()
        

    def forward(self, x, output_new):
        
        if self.case == 'case1': #
            
            output_new3 = self.new3(output_new)
            output_new3 = self.activation_new1(output_new3)

            output = (output_new3 + x) / 2.
            output = self.activation_new2(output)
            output = self.residual_module(output)

            return output, output_new3
            

        elif self.case == 'case2' or self.case == 'case7': #
            
            output_new3 = self.new3(x)
            output_new3 = self.activation_new1(output_new3)

            output = output_new3
            output = self.residual_module(output)

            return output, output_new3
        
        elif self.case == 'case3' or self.case == 'case6': #
            
            output_new3 = self.new3(output_new)
            output_new3 = self.activation_new1(output_new3)

            output = (output_new3 + x) / 2.
            output = self.activation_new2(output)

            return output, output_new3

        elif self.case == 'case4': #
            
            output_new3 = self.new3(x)
            output_new3 = self.activation_new1(output_new3)

            output = output_new3
            
            return output, output_new3
        
        elif self.case == 'case5': #
            
            output_new3 = self.new3(x)
            output_new3 = self.activation_new1(output_new3)

            output = output_new3
            
            return output, output_new3
        
        else:

            output_new3 = self.new3(output_new)
            output_new3 = self.activation_new1(output_new3)

            output = (output_new3 + x) / 2.
            output = self.activation_new2(output)
            output = self.residual_module(output)

            return output, output_new3
    

class Q1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Q1, self).__init__()
        
        self.mask = torch.from_numpy(np.array([[1,1,0],[1,0,0],[0,0,0]], dtype=np.float32)).cuda()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, padding = 1, kernel_size = 3)
        
    def forward(self, x):
        self.conv1.weight.data =  self.conv1.weight * self.mask
        x = self.conv1(x)
        
        return x        

class Q2(nn.Module):
    def __init__(self, in_ch, out_ch, dilated_value):
        super(Q2, self).__init__()
        
        self.mask = torch.from_numpy(np.array([[1,1,1],[1,1,0],[1,0,0]], dtype=np.float32)).cuda()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size = 3, padding=dilated_value, dilation=dilated_value)
        
    def forward(self, x):
        self.conv1.weight.data =  self.conv1.weight * self.mask
        x = self.conv1(x)
        
        return x        
    
class E1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(E1, self).__init__()
        
        self.mask = torch.from_numpy(np.array([[0,1,1],[0,0,1],[0,0,0]], dtype=np.float32)).cuda()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, padding = 1, kernel_size = 3)
        
    def forward(self, x):
        self.conv1.weight.data =  self.conv1.weight * self.mask
        x = self.conv1(x)
        
        return x        

class E2(nn.Module):
    def __init__(self, in_ch, out_ch, dilated_value):
        super(E2, self).__init__()
        
        self.mask = torch.from_numpy(np.array([[1,1,1],[0,1,1],[0,0,1]], dtype=np.float32)).cuda()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size = 3, padding=dilated_value, dilation=dilated_value)
        
    def forward(self, x):
        self.conv1.weight.data =  self.conv1.weight * self.mask
        x = self.conv1(x)
        
        return x        
    
class D1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D1, self).__init__()
        
        self.mask = torch.from_numpy(np.array([[0,0,0],[0,0,0],[1,1,1]], dtype=np.float32)).cuda()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, padding = 1, kernel_size = 3)
        
    def forward(self, x):
        self.conv1.weight.data =  self.conv1.weight * self.mask
        x = self.conv1(x)
        
        return x        

class D2(nn.Module):
    def __init__(self, in_ch, out_ch, dilated_value):
        super(D2, self).__init__()
        
        self.mask = torch.from_numpy(np.array([[0,0,0],[1,1,1],[1,1,1]], dtype=np.float32)).cuda()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size = 3, padding=dilated_value, dilation=dilated_value)
        
    def forward(self, x):
        self.conv1.weight.data =  self.conv1.weight * self.mask
        x = self.conv1(x)
        
        return x 
    
class QED_first_layer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(QED_first_layer, self).__init__()
        
        self.q1 = Q1(in_ch,out_ch)
        self.e1 = E1(in_ch,out_ch)
        self.d1 = D1(in_ch,out_ch)

    def forward(self, x):
        
        outputs = []
        
        outputs.append(self.q1(x))
        outputs.append(self.e1(x))
        outputs.append(self.d1(x))
        
        return outputs  
   
class QED_layer(nn.Module):
    def __init__(self, in_ch, out_ch, dilated_value):
        super(QED_layer, self).__init__()
        
        self.q2_prelu = nn.PReLU(in_ch,0).cuda()
        self.e2_prelu = nn.PReLU(in_ch,0).cuda()
        self.d2_prelu = nn.PReLU(in_ch,0).cuda()
        
        self.q2 = Q2(in_ch, out_ch, dilated_value)
        self.e2 = E2(in_ch, out_ch, dilated_value)
        self.d2 = D2(in_ch, out_ch, dilated_value)

    def forward(self, inputs):
        
        outputs = []

        out_q2 = self.q2_prelu(inputs[0])
        out_e2 = self.e2_prelu(inputs[1])
        out_d2 = self.d2_prelu(inputs[2])
        
        outputs.append(self.q2(out_q2))
        outputs.append(self.e2(out_e2))
        outputs.append(self.d2(out_d2))
        
        return outputs
