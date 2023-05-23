from __future__ import print_function, division
import os
import torch
import numpy        as np
import pandas       as pd
from torch.utils.data           import Dataset
from geotnf.transformation      import GeometricTnf
from geotnf.point_tnf           import PointTnf
from skimage                    import io
from torch.autograd             import Variable


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
        mri_x_array  = [int(float(a)) for a in self.MRI_x[idx].split(';')]
        mri_y_array  = [int(float(a)) for a in self.MRI_y[idx].split(';')]

        num_landmarks   = len(histo_x_array)
        if self.batch_size > 1:
            length = 26
        else:
            length = num_landmarks
            
        # Use the landmark coordinates to create histo 'landmarks image'
        landmarks_histo = np.zeros((image_histo.shape[0], image_histo.shape[1], length))
        landmarks_mri   = np.zeros((image_MRI.shape[0], image_MRI.shape[1], num_landmarks))
        
        # Make square to ensure resized image still contains landmark
        # Use 5x5 for histo and 3x3 for mri, since different amount of resizing
        for i in range(num_landmarks):
            x, y = (histo_x_array[i],histo_y_array[i])
            landmarks_histo[x-2:x+2,y-2:y+2,i] = 1
            x, y = (mri_x_array[i],mri_y_array[i])
            landmarks_mri[x-1:x+1,y-1:y+1,i] = 1
        
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
            
        # For the target, save landmark locations as array
        landmarks_mri = landmarks_mri.transpose(0,1).transpose(1,2)
         
        mri_landmark_list = np.zeros((2, length))
        for i in range(num_landmarks):
            (x,y) = np.unravel_index(np.argmax(landmarks_mri[:,:,i], axis=None), landmarks_mri[:,:,i].shape)  
            mri_landmark_list[:,i] = [x,y]
            
        sample = {'source_image': image_histo, 'target_image': image_MRI, 'landmarks_source': landmarks_histo, 'landmarks_target':mri_landmark_list}

        if self.transform:
            sample = self.transform(sample)

        return sample