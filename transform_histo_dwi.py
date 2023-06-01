import json
from process_img import * 
from geotnf.transformation import GeometricTnf
from skimage import io
from glob import glob
from register_functions import output_results
import torch
import gc

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

def transform_histo(transforms, path, use_cuda=True, out_size = 1024):
    gc.collect()
    torch.cuda.empty_cache()
    
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
        json_path_histo_t2  = './transforms/transform_' + sid + '.json'
        json_path_t2_dwi    = './transforms/transform_' + sid + '_T2_DWI.json'
        img_path   = os.path.join('./results/preprocess/hist' , sid+'_high_res/')
        
        # Get image paths
        all_paths = {}
        all_paths['histo'] = sorted(glob(img_path + 'hist*.png'))
        all_paths['mask']  = sorted(glob(img_path + 'mask*.png'))
        
        json_data_histo_t2 = get_data(json_path_histo_t2)
        json_data_t2_dwi   = get_data(json_path_t2_dwi)
        
        # Get sid params
        origin      = json_data_t2_dwi["origin"]
        direction   = json_data_t2_dwi["direction"]
        spacing     = json_data_t2_dwi["spacing"]
        
        t2_scale    = json_data_histo_t2["scale"] # This is 1
        dwi_scale   = json_data_t2_dwi["scale"]
        
        scale = 1024/80
        hist_space  = [spacing[0]/scale, spacing[1]/scale, spacing[2]]
        
        spatialInfo = (origin, hist_space, direction)
        
        count    = len(all_paths['histo'])
        out_size = 1024
        
        warped_imgs = {}
        for annot in all_paths:
            warped_imgs[annot] = np.zeros((count, out_size,out_size, 3))
        
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
                warped_imgs[annot][i,:,:,:] = transform_histo(transforms, all_paths[annot][i], use_cuda=True)
            
            # Apply mask to image
            mask = np.zeros((out_size, out_size, 3), dtype=int)
            for n in range(3):
                mask[:, :, n]  = warped_imgs['mask'][i, :, :, 0] 
            warped_imgs['histo'][i,:,:,:] = np.multiply(warped_imgs['histo'][i,:,:,:],mask)
            
        # Save images
        for annot in all_paths:
            output_results('./results/registration/histo-DWI/', warped_imgs[annot], sid, annot, spatialInfo, 'final', extension = '.nii.gz')
    
main()
