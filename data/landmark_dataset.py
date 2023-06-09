from __future__ import print_function, division
import os
import cv2
import torch
import numpy        as np
import pandas       as pd
from torch.utils.data           import Dataset
from geotnf.transformation      import GeometricTnf
from skimage                    import io
from torch.autograd             import Variable


def list_to_image(dims, x_array, y_array, m, tensor=False, use_cuda=True):
    # Use the landmark coordinates to create histo 'landmarks image'
    num_landmarks   = len(x_array)
    landmarks       = np.zeros((dims[0], dims[1], num_landmarks))
    
    # Make square to ensure resized image still contains landmark
    # Use 5x5 for histo and 3x3 for mri, since different amount of resizing
    for i in range(num_landmarks):
        x, y = (x_array[i],y_array[i])
        landmarks[int(x-m):int(x+m),int(y-m):int(y+m),i] = 1
        
    if tensor:
        # make arrays float tensor for subsequent processing
        landmarks = torch.Tensor(landmarks.astype(np.float32))
        # permute order of image to CHW
        landmarks = landmarks.transpose(1,2).transpose(0,1)
        landmarks = Variable(landmarks.unsqueeze(0),requires_grad=False)
        if use_cuda:
            landmarks = landmarks.cuda()
            
    return landmarks
            
def image_to_list(landmarks, use_cuda=False):
    if use_cuda:
        landmarks = torch.squeeze(landmarks).cpu().detach()
        
    num_landmarks = landmarks.shape[0]
    landmarks     = landmarks.transpose(0,1).transpose(1,2) # Shape is now 240 x 240 x N
    landmark_list = np.zeros((2,num_landmarks))
           
    # Find non-zero locations. Average over all of them. 
    for i in range(num_landmarks):
        non_zero           = np.argwhere(landmarks[:,:,i]).double()
        x, y               = torch.ceil(torch.mean(non_zero,axis=1))
        landmark_list[:,i] = [int(x),int(y)]
      
    landmark_list = torch.Tensor(landmark_list.astype(np.float32)) 
    
    if use_cuda:
        landmark_list = landmark_list.cuda()
       
    return landmark_list


class LandmarkDataset(Dataset):
    """    
    Args:
            csv_file (string): Path to the csv file with image names and landmarks.
            training_image_path (string): Directory with all the images.
            transform (callable): Transformation for post-processing the training pair (eg. image normalization)
            
    Returns:
            Dict: {'histo': histo image, 'MRI': mri image, 'source_landmarks': landmark location on histo, 'target_landmarks': landmark location on mri}
            
    """

    def __init__(self, csv_file, training_image_path, batch_size, output_size=(240,240), geometric_model='tps', transform=None,
                 random_sample=False, random_t=0.5, random_s=0.5, random_alpha=1/6, random_t_tps=0.4, use_cuda=True):
        
        # random_sample is used to indicate whether deformation coefficients are randomly generated?
        self.random_sample      = random_sample
        self.random_t           = random_t
        self.random_t_tps       = random_t_tps
        self.random_alpha       = random_alpha
        self.random_s           = random_s
        self.out_h, self.out_w  = output_size
        
        # read csv file
        self.train_data         = pd.read_csv(csv_file)
        self.img_histo_names    = self.train_data.iloc[:,0]
        self.img_MRI_names      = self.train_data.iloc[:,1]
        self.histo_y            = self.train_data.iloc[:,2]
        self.histo_x            = self.train_data.iloc[:,3]
        self.MRI_y              = self.train_data.iloc[:,4]
        self.MRI_x              = self.train_data.iloc[:,5]
        
        # copy arguments
        self.training_image_path    = training_image_path
        self.transform              = transform
        self.geometric_model        = geometric_model
        self.use_cuda               = use_cuda
        self.batch_size             = batch_size
        
        # affine transform used to rescale images
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda = False) 
        
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        # read image
        img_histo_name  = os.path.join(self.training_image_path, self.img_histo_names[idx])
        image_histo     = io.imread(img_histo_name)

        img_MRI_name  = os.path.join(self.training_image_path, self.img_MRI_names[idx])
        MRI           = io.imread(img_MRI_name)
        image_MRI     = np.zeros((MRI.shape[0],MRI.shape[1],3))
        for i in range(3):
            image_MRI[:,:,i] = MRI
        
        # Get the landmarks in the correct format - as an int list
        histo_x_array = [int(float(a)) for a in self.histo_x[idx].split(';')]
        histo_y_array = [int(float(a)) for a in self.histo_y[idx].split(';')]
        mri_x_array   = [int(float(a)) for a in self.MRI_x[idx].split(';')]
        mri_y_array   = [int(float(a)) for a in self.MRI_y[idx].split(';')]

        num_landmarks   = len(histo_x_array)
        if self.batch_size > 1:
            length = 26
        else:
            length = num_landmarks
            
        landmarks_histo = list_to_image(image_histo.shape[:2], histo_x_array, histo_y_array, 2)
        landmarks_mri   = list_to_image(image_MRI.shape[:2],   mri_x_array, mri_y_array, 1)
        
        # Crop HMU_025_SH
        if '25' in self.img_MRI_names[idx]:
            image_MRI       = image_MRI[160:260,160:290,:]
            landmarks_mri   = landmarks_mri[160:260,160:290,:]
            image_histo     = image_histo[:,160:,:]
            landmarks_histo = landmarks_histo[:,160:,:]
            
        # Pad MRI & MRI landmarks to make square
        if MRI.shape[0] > MRI.shape[1]:
            pad_h = 20
            pad_w = int((MRI.shape[0]-MRI.shape[1])/2) + 20
        else:
            pad_h = int((MRI.shape[1]-MRI.shape[0])/2) + 20
            pad_w = 20

        image_MRI       = cv2.copyMakeBorder(image_MRI, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT,value=0)
        landmarks_mri   = cv2.copyMakeBorder(landmarks_mri, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT,value=0)
        
        image_histo     = cv2.copyMakeBorder(image_histo,     100, 100, 100, 100, cv2.BORDER_CONSTANT, value=0)
        landmarks_histo = cv2.copyMakeBorder(landmarks_histo, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=0)
        
        # make arrays float tensor for subsequent processing
        image_histo     = torch.Tensor(image_histo.astype(np.float32))
        image_MRI       = torch.Tensor(image_MRI.astype(np.float32))
        landmarks_histo = torch.Tensor(landmarks_histo.astype(np.float32))
        landmarks_mri   = torch.Tensor(landmarks_mri.astype(np.float32))

        # permute order of image to CHW
        image_histo     = image_histo.transpose(1,2).transpose(0,1)
        image_MRI       = image_MRI.transpose(1,2).transpose(0,1)
        landmarks_histo = landmarks_histo.transpose(1,2).transpose(0,1)
        landmarks_mri   = landmarks_mri.transpose(1,2).transpose(0,1)
                
        # Resize image using bilinear sampling with identity affine tnf
        if image_histo.size()[0]!=self.out_h or image_histo.size()[1]!=self.out_w:
            image_histo     = self.affineTnf(Variable(image_histo.unsqueeze(0),requires_grad=False)).data.squeeze(0)
            landmarks_histo = self.affineTnf(Variable(landmarks_histo.unsqueeze(0),requires_grad=False)).data.squeeze(0)
            
        if image_MRI.size()[0]!=self.out_h or image_MRI.size()[1]!=self.out_w:
            image_MRI     = self.affineTnf(Variable(image_MRI.unsqueeze(0),requires_grad=False)).data.squeeze(0)
            landmarks_mri = self.affineTnf(Variable(landmarks_mri.unsqueeze(0),requires_grad=False)).data.squeeze(0)
            
        # Save landmark locations as array
        histo_landmark_list = image_to_list(landmarks_histo)
        mri_landmark_list   = image_to_list(landmarks_mri)
            
        sample = {'source_image': image_histo, 'target_image': image_MRI, 'landmarks_source': histo_landmark_list, 'landmarks_target':mri_landmark_list, 'name':self.img_histo_names[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample