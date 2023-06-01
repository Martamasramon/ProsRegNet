
import torch
from image.normalization import normalize_image
from geotnf.transformation import GeometricTnf
from geotnf.point_tnf import *
import sys
sys.path.insert(0, '../parse_data/parse_json')
import numpy as np
from preprocess import *

def preprocess_image(image, out_size=240):
    """ 
    Normalise image 
    """
    
    resizeCNN = GeometricTnf(out_h=out_size, out_w=out_size, use_cuda = False) 

    # convert to torch Variable
    image       = np.expand_dims(image.transpose((2,0,1)),0)
    image       = torch.Tensor(image.astype(np.float32)/255.0)
    image_var   = Variable(image,requires_grad=False)

    # Resize image using bilinear sampling with identity affine tnf
    image_var = resizeCNN(image_var)

    # Normalize image
    image_var = normalize_image(image_var)

    return image_var


def process_image(input_image, use_cuda, out_size=240, mask=False):
    if mask:
        image = np.copy(input_image)
        image[np.any(image > 5, axis=-1)] = 255
    else:
        image = input_image
    
    image_var = preprocess_image(image, out_size=out_size)
        
    if use_cuda:
        image_var = image_var.cuda()
        
    return image_var
    