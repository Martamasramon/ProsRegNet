from __future__ import print_function, division
import torch
from geotnf.transformation      import GeometricTnf
from torch.autograd             import Variable

class LandmarkTnf(object):
    
    def __init__(self, use_cuda=True):
        assert isinstance(use_cuda, (bool))
        self.use_cuda           = use_cuda
        
    def __call__(self, batch):
        image_batch_A, image_batch_B = batch['source_image'], batch['target_image']
        landmarks_A, landmarks_B     = batch['landmarks_source'], batch['landmarks_target']
        
        if self.use_cuda:
            image_batch_A = image_batch_A.cuda()
            image_batch_B = image_batch_B.cuda()
            landmarks_A   = landmarks_A.cuda()
            landmarks_B   = landmarks_B.cuda()
        
        # convert to variables
        histo_image_batch   = Variable(image_batch_A,requires_grad=False)
        mri_image_batch     = Variable(image_batch_B,requires_grad=False)
        histo_landmarks     = Variable(landmarks_A,requires_grad=False)
        mri_landmarks       = Variable(landmarks_B,requires_grad=False)

        Ones_A  = torch.ones(histo_image_batch.size())
        Zeros_A = torch.zeros(histo_image_batch.size())
        Ones_B  = torch.ones(mri_image_batch.size())
        Zeros_B = torch.zeros(mri_image_batch.size())
        
        if self.use_cuda:
            Ones_A  = Ones_A.cuda()
            Zeros_A = Zeros_A.cuda()
            Ones_B  = Ones_B.cuda()
            Zeros_B = Zeros_B.cuda()
            
        histo_mask_batch = torch.where(histo_image_batch != 0.0 *Ones_A, Ones_A, Zeros_A)
        mri_mask_batch   = torch.where(mri_image_batch   != 0.0 *Ones_B, Ones_B, Zeros_B)
        
        if self.use_cuda:
            histo_image_batch   = histo_image_batch.cuda()
            mri_image_batch     = mri_image_batch.cuda()
            histo_mask_batch    = histo_mask_batch.cuda()
            mri_mask_batch      = mri_mask_batch.cuda()
            histo_landmarks     = histo_landmarks.cuda()
            mri_landmarks       = mri_landmarks.cuda()

        return {'source_image':     histo_image_batch,  'target_image':     mri_image_batch, 
                'source_mask':      histo_mask_batch,   'target_mask':      mri_mask_batch, 
                'source_landmarks': histo_landmarks,    'target_landmarks': mri_landmarks,
                'name':             batch['name']}
