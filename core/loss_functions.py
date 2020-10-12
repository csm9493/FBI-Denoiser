import torch
import torch.nn as nn

def mse_bias(output, target):
    
    X = output
    b = target
    
    # E[(X - b)**2]
    loss = torch.mean((target - output)**2)
    
    return loss

def estimated_bias(output, target):
    
    Z = target[:,0]
    b = output
    sigma = target[:,1]
    
    # E[(Z - b)**2 - sigma**2]
    loss = torch.mean((Z - b)**2 - sigma**2)
    
    return loss

def mse_affine(output, target):
    
    a = output[:,0]
    b = output[:,1]
    Z = target[:,0]
    X = target[:,1]
    
    # E[(X - (aZ+b))**2]
    loss = torch.mean((X - (a*Z+b))**2)
    
    return loss

def estimated_affine(output, target):
    
    a = output[:,0]
    b = output[:,1]
    
    Z = target[:,0]
    sigma = target[:,1]
    # E[(Z - (aZ+b))**2 + 2a(sigma**2) - sigma**2]
    loss = torch.mean((Z - (a*Z+b))**2 + 2*a*(sigma**2) - sigma**2)
    
    return loss
