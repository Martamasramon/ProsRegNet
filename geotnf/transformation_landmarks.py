from __future__ import print_function, division
import torch
from geotnf.transformation      import GeometricTnf
from torch.autograd             import Variable

class LandmarkTnf(object):
    
    def __init__(self, use_cuda=True, crop_factor=16/16, output_size=(240,240), padding_factor = 0.0):
        assert isinstance(use_cuda, (bool))
        assert isinstance(crop_factor, (float))
        assert isinstance(output_size, (tuple))
        assert isinstance(padding_factor, (float))
        self.use_cuda           = use_cuda
        self.crop_factor        = crop_factor
        self.padding_factor     = padding_factor
        self.out_h, self.out_w  = output_size 
        self.rescalingTnf = GeometricTnf('affine', self.out_h, self.out_w, use_cuda = self.use_cuda)
        
    def __call__(self, batch):
        image_batch_A, image_batch_B = batch['source_image'], batch['target_image']
        landmarks_A, landmarks_B     = batch['landmarks_source'], batch['landmarks_target']
        
        if self.use_cuda:
            image_batch_A = image_batch_A.cuda()
            image_batch_B = image_batch_B.cuda()
            landmarks_A   = landmarks_A.cuda()
            landmarks_B   = landmarks_B.cuda()
                          
        # generate symmetrically padded image for bigger sampling region
        #image_batch_A = self.symmetricImagePad(image_batch_A,self.padding_factor)
        #image_batch_B = self.symmetricImagePad(image_batch_B,self.padding_factor)
        
        # convert to variables
        histo_image_batch   = Variable(image_batch_A,requires_grad=False)
        mri_image_batch     = Variable(image_batch_B,requires_grad=False)
        histo_landmarks     = Variable(landmarks_A,requires_grad=False)
        mri_landmarks       = Variable(landmarks_B,requires_grad=False)

        # get cropped image
        #cropped_image_batch = self.rescalingTnf(image_batch_A,None,self.padding_factor,self.crop_factor) # Identity is used as no theta given
        
        Ones_A  = torch.ones(histo_image_batch.size())
        Zeros_A = torch.zeros(histo_image_batch.size())
        Ones_B  = torch.ones(mri_image_batch.size())
        Zeros_B = torch.zeros(mri_image_batch.size())
        
        if self.use_cuda:
            Ones_A  = Ones_A.cuda()
            Zeros_A = Zeros_A.cuda()
            Ones_B  = Ones_B.cuda()
            Zeros_B = Zeros_B.cuda()
            
        histo_mask_batch = torch.where(histo_image_batch > 0.1*Ones_A, Ones_A, Zeros_A)
        mri_mask_batch   = torch.where(mri_image_batch   > 0.1*Ones_B, Ones_B, Zeros_B)
        
        if self.use_cuda:
            histo_image_batch   = histo_image_batch.cuda()
            mri_image_batch     = mri_image_batch.cuda()
            histo_mask_batch    = histo_mask_batch.cuda()
            mri_mask_batch      = mri_mask_batch.cuda()
            histo_landmarks     = histo_landmarks.cuda()
            mri_landmarks       = mri_landmarks.cuda()
        
        #mask1 = 255*normalize_image(warped_mask_batch,forward=False)
        #mask1 = mask1.data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()
        
        #print(mask1.shape)

        #io.imsave('warped_mask.jpg', mask1)

        return {'source_image':     histo_image_batch,  'target_image':     mri_image_batch, 
                'source_mask':      histo_mask_batch,   'target_mask':      mri_mask_batch, 
                'source_landmarks': histo_landmarks,    'target_landmarks': mri_landmarks}
