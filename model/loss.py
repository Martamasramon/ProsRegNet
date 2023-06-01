from __future__ import print_function, division
import torch
import torch.nn     as nn
import numpy        as np
from torch.autograd         import Variable
from geotnf.point_tnf       import PointTnf
from geotnf.transformation  import GeometricTnf
from skimage                import io

class SSDLoss(nn.Module):
    def __init__(self, use_cuda=True, geometric_model='affine', out_size=240):
        super(SSDLoss, self).__init__()
        self.geometric_model    = geometric_model
        self.use_cuda           = use_cuda
        self.out_h, self.out_w  = (out_size, out_size)

    def forward(self, theta, tnf_batch):
        ### compute square root of ssd
        geometricTnf = GeometricTnf(self.geometric_model, out_h=self.out_h, out_w=self.out_w, use_cuda = self.use_cuda)
        
        A = tnf_batch['target_image']
        B = geometricTnf(tnf_batch['source_image'],theta)
        
        ssd = torch.sum(torch.sum(torch.sum(torch.pow(A - B,2),dim=3),dim=2),dim=1)
        ssd = torch.sum(ssd)/(A.shape[0]*A.shape[1]*A.shape[2]*A.shape[3])
        ssd = torch.sqrt(ssd)
        
        return  ssd 

class MSELoss(nn.Module):
    def __init__(self, use_cuda=True):
        super(MSELoss, self).__init__()
        self.use_cuda   = use_cuda
        self.pointTnf   = PointTnf(use_cuda=self.use_cuda)

    def forward(self, theta, tnf_batch):
        # A & B shape is [B,2,N]
        A = tnf_batch['target_landmarks']
        B = self.pointTnf.tpsPointTnf(theta, tnf_batch['source_landmarks'])
        
        mse = torch.mean((A - B)*(A - B))
        
        return mse
        