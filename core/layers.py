import torch
import torch.nn as nn
import numpy as np

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
    def __init__(self, in_ch):
        super(Residual_module, self).__init__()
        
        self.activation1 = nn.PReLU(in_ch,0).cuda()
        self.activation2 = nn.PReLU(in_ch,0).cuda()
            
        self.conv1_1by1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size = 1)
        self.conv2_1by1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size = 1)

    def forward(self, input):

        output_residual = self.activation1(input)
        output_residual = self.conv1_1by1(output_residual)
        output_residual = self.activation2(output_residual)
        output_residual = self.conv2_1by1(output_residual)
        
        output = input + output_residual
        
        return output
    
class New1_layer(nn.Module):
    def __init__(self, in_ch, out_ch, case = 'final'):
        super(New1_layer, self).__init__()
        
        self.new1 = New1(in_ch,out_ch).cuda()
        if case != 'case1':
            self.residual_module = Residual_module(out_ch)
        
        self.case = case

    def forward(self, x):
        
        
        if self.case == 'case1': # plain NN architecture wo residual module and residual connection
            
            output_new1 = self.new1(x)
            
            return output_new1
        
        elif self.case == 'case2': # plain NN architecture wo residual connection
            
            output_new1 = self.new1(x)
            output = self.residual_module(output_new1)
            
            return output
            
        else: # final model
        
            output_new1 = self.new1(x)
            output = self.residual_module(output_new1)

            return output, output_new1
   
class New2_layer(nn.Module):
    def __init__(self, in_ch, out_ch, case = 'final'):
        super(New2_layer, self).__init__()
        
        self.new2 = New2(in_ch,out_ch).cuda()
        if case != 'case1':
            self.residual_module = Residual_module(out_ch)
        self.activation_new2 = nn.PReLU(in_ch,0).cuda()
        
        self.case = case

    def forward(self, x, output_new):
        
        if self.case == 'case1': # plain NN architecture wo residual module and residual connection
            
            output_new2 = self.activation_new2(x)
            output_new2 = self.new2(output_new2)

            return output_new2
        
        elif self.case == 'case2': # plain NN architecture wo residual connection
            
            output_new2 = self.activation_new2(x)
            output_new2 = self.new2(output_new2)
            output_new2 = self.residual_module(output_new2)

            return output_new2
        
        else:

            output_new2 = self.activation_new2(output_new)
            output_new2 = self.new2(output_new2)

            output = output_new2 + x
            output = self.residual_module(output)

            return output, output_new2
            
    
class New3_layer(nn.Module):
    def __init__(self, in_ch, out_ch, dilated_value=3, case = 'final'):
        super(New3_layer, self).__init__()
        
        self.new3 = New3(in_ch,out_ch,dilated_value).cuda()
        if case != 'case1':
            self.residual_module = Residual_module(out_ch)
        self.activation_new3 = nn.PReLU(in_ch,0).cuda()
        
        self.case = case

    def forward(self, x, output_new):
        
        if self.case == 'case1':
            
            output_new3 = self.activation_new3(x)
            output_new3 = self.new3(output_new3)

            return output_new3
            
        elif self.case == 'case2':
            
            output_new3 = self.activation_new3(x)
            output_new3 = self.new3(output_new3)
            output_new3 = self.residual_module(output_new3)

            return output_new3
        
        else: 

            output_new3 = self.activation_new3(output_new)
            output_new3 = self.new3(output_new3)

            output = output_new3 + x
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
    
class Average_layer(nn.Module):
    def __init__(self, in_ch, activation = 'PReLU'):
        super(Average_layer, self).__init__()
        
        if activation == 'ReLU':
            self.activation = nn.ReLU().cuda()
        else:
            self.activation = nn.PReLU(in_ch,0).cuda()

    def forward(self, inputs):

        mean = torch.mean(torch.stack(inputs), dim=0)
        output = self.activation(mean)
        
        return output