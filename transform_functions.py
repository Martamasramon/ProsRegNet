import json
from process_img import * 
from geotnf.transformation import GeometricTnf
from skimage import io
from glob import glob
from register_functions import output_results
import torch
import argparse
import cv2

def get_data(filename): 
    with open(filename) as file:
        file_contents = file.read()
    
    json_data = json.loads(file_contents)
    
    return json_data
    
def create_tensor(transform, use_cuda=True):
    transform = torch.Tensor(transform)
    
    if use_cuda:
        transform = transform.cuda()
        
    return transform

def resize_from_coord(coord, sid, idx, x_s, y_s):
    x_prime         = coord[sid]['x'][idx]
    y_prime         = coord[sid]['y'][idx]
    x_offset_prime  = coord[sid]['x_offset'][idx]
    y_offset_prime  = coord[sid]['y_offset'][idx]
    h_prime         = coord[sid]['h'][idx]
    w_prime         = coord[sid]['w'][idx]
    
    w_new = int(w_prime * x_s) + 2*int(x_offset_prime * x_s)
    h_new = int(h_prime * y_s) + 2*int(y_offset_prime * y_s)
    
    start_x = int(x_prime * x_s) - int(x_offset_prime * x_s)
    end_x   = int(x_prime * x_s) + int(w_prime * x_s) + int(x_offset_prime * x_s)
    start_y = int(y_prime * y_s) - int(y_offset_prime * y_s)
    end_y   = int(y_prime * y_s) + int(h_prime * y_s) + int(y_offset_prime * y_s)
    
    return (h_new, w_new), start_x, end_x, start_y, end_y
        
      
def transform_histo(transforms, path, flip_v, flip_h, use_cuda=True, out_size = 1024):
    """ 
    Apply transformations to histology: 2 affine + 1 TPS 
    """
    
    # Get transforms
    theta_aff_1, theta_aff_2, theta_tps = transforms
        
    # Load geometric models
    affTnf      = GeometricTnf(geometric_model='affine', out_h=out_size, out_w=out_size, use_cuda=use_cuda)
    tpsTnf      = GeometricTnf(geometric_model='tps',    out_h=out_size, out_w=out_size, use_cuda=use_cuda)
    
    # Preprocess image 
    source_image     = cv2.imread(path)
    if flip_h:
        source_image = cv2.flip(source_image,1)
    source_image_var = process_image(source_image, use_cuda, out_size=out_size)

    # Apply transformations 
    warped_image = affTnf(source_image_var, theta_aff_1.view(-1,2,3))
    warped_image = affTnf(warped_image, theta_aff_2.view(-1,2,3))
    warped_image = tpsTnf(warped_image,theta_tps)
     
    # Un-normalize images and convert to numpy
    warped_image_np = normalize_image(warped_image,forward=False).data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()
    
    # Ignore negative values
    warped_image_np[warped_image_np < 0] = 0 

    return warped_image_np

  
def transform_histo_double(transforms, path, use_cuda=True, out_size = 1024):
    """ 
    Apply transformations to histology: (2 affine + 1 TPS) * 2
    Concatenate transformations found for histo-T2 and T2-DWI.
    """
    
    # Get transforms
    theta_aff_1, theta_aff_2, theta_tps_1, theta_aff_3, theta_aff_4, theta_tps_2 = transforms
        
    # Load geometric models
    affTnf      = GeometricTnf(geometric_model='affine', out_h=out_size, out_w=out_size, use_cuda=use_cuda)
    tpsTnf      = GeometricTnf(geometric_model='tps',    out_h=out_size, out_w=out_size, use_cuda=use_cuda)
    tpsMRITnf   = GeometricTnf(geometric_model='tps-mri',out_h=out_size, out_w=out_size, use_cuda=use_cuda)
    
    # Preprocess image 
    source_image     = io.imread(path)
    source_image_var = process_image(source_image, use_cuda, out_size=out_size)

    # Apply transformations 
    warped_image = affTnf(source_image_var, theta_aff_1.view(-1,2,3))
    warped_image = affTnf(warped_image, theta_aff_2.view(-1,2,3))
    warped_image = tpsTnf(warped_image,theta_tps_1)
    warped_image = affTnf(warped_image, theta_aff_3.view(-1,2,3))
    warped_image = affTnf(warped_image, theta_aff_4.view(-1,2,3))
    warped_image = tpsMRITnf(warped_image,theta_tps_2)
     
    # Un-normalize images and convert to numpy
    warped_image_np = normalize_image(warped_image,forward=False).data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()
    
    # Ignore negative values
    warped_image_np[warped_image_np < 0] = 0 
       
   
    return warped_image_np
