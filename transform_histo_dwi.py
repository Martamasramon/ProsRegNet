import json
from process_img import * 
from geotnf.transformation import GeometricTnf
from skimage import io
from glob import glob
from register_functions import output_results
import torch
import gc
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
        
        
def transform_histo(transforms, path, use_cuda=True, out_size = 1024):
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

samples = ['HMU_003_DB', 'HMU_007_TN', 'HMU_010_FH']

def main():
    for sid in samples:
        # Define resolution
        half_out_size = 512
        
        # Get info from json files
        json_path_histo_t2  = './transforms/transform_' + sid + '.json'
        json_path_t2_dwi    = './transforms/transform_' + sid + '_T2_DWI.json'
        json_data_histo_t2  = get_data(json_path_histo_t2)
        json_data_t2_dwi    = get_data(json_path_t2_dwi)
        
        # Get image paths
        img_path   = os.path.join('./results/preprocess/hist' , sid+'_high_res/')
        all_paths = {}
        all_paths['_histo'] = sorted(glob(img_path + 'hist*.png'))
        all_paths['_mask']  = sorted(glob(img_path + 'mask*.png'))
        count               = len(all_paths['_histo'])
        
        # Get coordinate information of dwi
        with open('coord_dwi.txt') as f:
            coord = json.load(f)
        slices = coord[sid]['slice']
        
        dwi_path        = os.path.join('./results/preprocess/dwi/' , sid, 'mriUncropped_' + sid + '_' + str(slices[0]).zfill(2) + '.jpg')
        w, h, _         = (cv2.imread(dwi_path)).shape
        padding_factor  = int(round(max(np.add(coord[sid]['h'],np.multiply(2,coord[sid]['y_offset'])))/(coord[sid]['h'][0]+2*coord[sid]['y_offset'][0])))
        y_s             = (half_out_size*(2+2*padding_factor))/h
        x_s             = (half_out_size*(2+2*padding_factor))/w  
        #scale = 2*half_out_size/80
                    
        # Get sid params
        origin      = json_data_t2_dwi["origin"]        
        direction   = json_data_t2_dwi["direction"]    
        spacing     = json_data_t2_dwi["spacing"]       
        
        hist_space  = [spacing[0]/x_s, spacing[1]/y_s, spacing[2]]
        spatialInfo = (origin, hist_space, direction)
        
        warped_imgs = {}
        out_histo   = {}
        for annot in all_paths:
            warped_imgs[annot] = np.zeros((count, 2*half_out_size, 2*half_out_size, 3))
            out_histo[annot]   = np.zeros((count, half_out_size*(2+2*padding_factor), half_out_size*(2+2*padding_factor), 3))

        # Get transformation params
        for i in range(count):
            theta_aff_1 = json_data_histo_t2[str(i)]["affine_1"]
            theta_aff_2 = json_data_histo_t2[str(i)]["affine_2"]
            theta_tps_1 = json_data_histo_t2[str(i)]["tps"]
            
            theta_aff_3 = json_data_t2_dwi[str(i)]["affine_1"]
            theta_aff_4 = json_data_t2_dwi[str(i)]["affine_2"]
            theta_tps_2 = json_data_t2_dwi[str(i)]["tps"]
            transforms  = (create_tensor(theta_aff_1), create_tensor(theta_aff_2), create_tensor(theta_tps_1), 
                            create_tensor(theta_aff_3), create_tensor(theta_aff_4), create_tensor(theta_tps_2))
            
            for annot in all_paths:
                warped_imgs[annot][i,:,:,:] = transform_histo(transforms, all_paths[annot][i], use_cuda=True, out_size=2*half_out_size)
            
            # Transform to MRI space
            new_size, start_x, end_x, start_y, end_y = resize_from_coord(coord, sid, i, x_s, y_s)
                        
            for annot in all_paths:
                out_histo[annot][i, start_x:end_x, start_y:end_y, :] = cv2.resize(warped_imgs[annot][i,:,:,:], new_size, interpolation=cv2.INTER_CUBIC)  

        # Save images
        for annot in all_paths:
            output_results('./results/registration/histo-DWI/', out_histo[annot], sid, annot, spatialInfo, 'final', extension = '.nii.gz')
    
main()
