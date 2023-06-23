from __future__ import print_function, division
import torch
import cv2
import os 
import numpy                as np
from geotnf.transformation  import GeometricTnf
from torch.autograd         import Variable
from preprocess             import *


def list_to_image(dims, x_array, y_array, m, tensor=False, use_cuda=True, compact=False):
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
            
    if compact:
        temp        = landmarks
        landmarks   = np.zeros((dims[0], dims[1], 3))
        for i in range(num_landmarks):
            landmarks[:,:,1] += temp[:,:,i]
    
    return landmarks
         
            
def image_to_list(landmarks, use_cuda=False, hist=True):
    if use_cuda:
        landmarks = torch.squeeze(landmarks).cpu().detach()
      
    if hist:
        landmarks = landmarks.transpose(0,1).transpose(1,2) # Shape is now 240 x 240 x N
    else:
        landmarks     = torch.Tensor(landmarks.astype(np.float32))
       
    num_landmarks = landmarks.shape[2]  
    landmark_list = np.zeros((2,num_landmarks))
            
    # Find non-zero locations. Average over all of them. 
    for i in range(num_landmarks):
        non_zero           = np.argwhere(landmarks[:,:,i]).double()
        x, y               = torch.ceil(torch.mean(non_zero,axis=1))
        x[np.isnan(x)]     = 0
        y[np.isnan(y)]     = 0
        landmark_list[:,i] = [int(x),int(y)]
      
    landmark_list = torch.Tensor(landmark_list.astype(np.float32)) 
    
    if use_cuda:
        landmark_list = landmark_list.cuda()
       
    return landmark_list


def get_landmark_img(dims, land_x, land_y, m=3, permute=True):
    # Get the landmarks in the correct format - as an int list
    x_array = [int(float(a)) for a in land_x.split(';')]
    y_array = [int(float(a)) for a in land_y.split(';')]
    
    # Transform to image -> Float tensor -> permute dims order
    landmarks = list_to_image(dims, y_array, x_array, m)
    landmarks = torch.Tensor(landmarks.astype(np.float32))
    
    if permute:
        landmarks = landmarks.transpose(1,2).transpose(0,1)
      
    return landmarks


def preprocess_landmarks(s, rows, cols, M, fIC, dims, size):
    print('Processing landmarks...')

    x, y, w, h, x_offset, y_offset = dims
    
    # Read landmarks
    land_x = s['landmarks-x']
    land_y = s['landmarks-y']
    landmarks  = get_landmark_img((rows,cols), land_x, land_y, permute=False)

    # Rotate & flip image 
    landmarks           = np.pad(landmarks,((rows,rows),(cols,cols),(0,0)),'constant', constant_values=0)
    rows, cols, _       = landmarks.shape
    rotated_landmarks   = cv2.warpAffine(landmarks,M,(cols,rows))
    
    if s['transform']['flip_v'] == 1: 
        rotated_landmarks   = cv2.flip(rotated_landmarks, 0)
    if s['transform']['flip_h'] == 1: 
        rotated_landmarks   = cv2.flip(rotated_landmarks, 1)
    if fIC: 
        rotated_landmarks   = cv2.flip(rotated_landmarks, 1)
        

    # Crop, pad & resize image
    crop    = rotated_landmarks[x:x+w, y:y+h,:]
    padded  = np.zeros((w + 2*x_offset, h + 2*y_offset, landmarks.shape[2])) 
    padded[x_offset:crop.shape[0]+x_offset, y_offset:crop.shape[1]+y_offset, :] = crop
    padded  = cv2.resize(padded, (size, size), interpolation=cv2.INTER_CUBIC)
    
    return padded
    
def preprocess_mri_landmarks(landmark_imgs, dims, case, idx, directory):
    
    min_x, x, w, x_offset, min_y, y, h, y_offset, upsHeight, upsWidth = dims
    
    for i in range(len(landmark_imgs)):
        try: 
            os.mkdir(directory + str(i) + '/')   
        except: 
            pass 
        
        im_slice = landmark_imgs[i][idx, :, :] * 255
        crop     = im_slice[min_x:x+w+x_offset, min_y:y+h +y_offset]
        upsample = cv2.resize(crop.astype('float32'), (upsHeight,  upsWidth), interpolation=cv2.INTER_CUBIC)
        
        cv2.imwrite(directory + str(i) + '/landmark_highRes' + str(i) + '_' + case + '_' + str(idx).zfill(2) +'.jpg', np.uint8(im_slice))
        cv2.imwrite(directory + str(i) + '/landmark_cropped' + str(i) + '_' + case + '_' + str(idx).zfill(2) +'.jpg', np.uint8(upsample))
                    
    
    
    
def transform_landmarks(landmarks, transforms, use_cuda=True, out_size=240):
    theta_aff_1, theta_aff_2, theta_tps = transforms
    
    # Convert to variable
    landmarks = torch.Tensor(landmarks.astype(np.float32)/255.0)
    landmarks = Variable(landmarks, requires_grad=False)
    landmarks = landmarks.transpose(1,2).transpose(0,1).unsqueeze(0)

    if use_cuda:
        landmarks = landmarks.cuda()
     
    # Get transformation models   
    affTnf = GeometricTnf(geometric_model='affine', out_h=out_size, out_w=out_size, use_cuda=use_cuda)
    tpsTnf = GeometricTnf(geometric_model='tps',    out_h=out_size, out_w=out_size, use_cuda=use_cuda)
    
    # Apply transformations
    warped_landmarks = affTnf(landmarks,        theta_aff_1.view(-1,2,3))
    warped_landmarks = affTnf(warped_landmarks, theta_aff_2.view(-1,2,3))
    warped_landmarks = tpsTnf(warped_landmarks, theta_tps)
    
    landmark_locations = image_to_list(warped_landmarks, use_cuda=use_cuda)
    
    return landmark_locations