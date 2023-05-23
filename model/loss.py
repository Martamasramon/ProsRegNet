from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from geotnf.point_tnf import PointTnf
from geotnf.transformation import GeometricTnf
from skimage import io

class SSDLoss(nn.Module):
    def __init__(self, geometric_model='affine', use_cuda=True):
        super(SSDLoss, self).__init__()
        self.geometric_model    = geometric_model
        self.use_cuda           = use_cuda

    def forward(self, theta, theta_GT, tnf_batch):
        ### compute square root of ssd
        geometricTnf = GeometricTnf(self.geometric_model, use_cuda = self.use_cuda)
        
        A = tnf_batch['target_image']
        B = geometricTnf(tnf_batch['source_image'],theta)
        
        ssd = torch.sum(torch.sum(torch.sum(torch.pow(A - B,2),dim=3),dim=2),dim=1)
        ssd = torch.sum(ssd)/(A.shape[0]*A.shape[1]*A.shape[2]*A.shape[3])
        ssd = torch.sqrt(ssd)
        
        return  ssd 

class MSELoss(nn.Module):
    def __init__(self, geometric_model='affine', use_cuda=True):
        super(MSELoss, self).__init__()
        self.geometric_model    = geometric_model
        self.use_cuda           = use_cuda

    def forward(self, theta, tnf_batch):
        # Assume batch_size = 1
        geometricTnf = GeometricTnf(self.geometric_model, use_cuda=self.use_cuda)

        A = tnf_batch['target_landmarks']
        B = geometricTnf(tnf_batch['source_landmarks'], theta)
        
        # Warped landmarks shape is [N,num_landmarks,240,240]
        # tnf_batch['target_landmarks'] shape is [N,num_landmarks,2]
        _,length,_ = A.shape
        
        temp        = B.view(length, -1).argmax(1).view(-1, 1)
        indices     = torch.cat((temp // 240, temp % 240), dim=1)
        # Indices has shape [num_landmarks,2]
        # Reshape A to same dimensions
        A = A.view(length, -1) 
    
        mse = torch.mean((A - indices)*(A - indices), dim=0)
        
        return mse
        