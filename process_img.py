
import torch
from image.normalization import normalize_image
from geotnf.transformation import GeometricTnf
from geotnf.point_tnf import *
import sys
sys.path.insert(0, '../parse_data/parse_json')
import numpy as np
from preprocess import *

def preprocess_image(image, half_out_size=512, high_res=False):
    """ 
    Normalise image 
    """
    
    if  high_res:
        resizeCNN = GeometricTnf(out_h=half_out_size*2, out_w=half_out_size*2, use_cuda = False) 
    else:
        resizeCNN = GeometricTnf(out_h=240, out_w=240, use_cuda = False) 

    # convert to torch Variable
    image       = np.expand_dims(image.transpose((2,0,1)),0)
    image       = torch.Tensor(image.astype(np.float32)/255.0)
    image_var   = Variable(image,requires_grad=False)

    # Resize image using bilinear sampling with identity affine tnf
    image_var = resizeCNN(image_var)

    # Normalize image
    image_var = normalize_image(image_var)

    return image_var


def process_image(input_image, use_cuda, half_out_size=512, high_res=False, mask=False):
    if mask:
        image = np.copy(input_image)
        image[np.any(image > 5, axis=-1)] = 255
    else:
        image = input_image
    
    image_var = preprocess_image(image, half_out_size=half_out_size, high_res=high_res)
        
    if use_cuda:
        image_var = image_var.cuda()
        
    return image_var
    