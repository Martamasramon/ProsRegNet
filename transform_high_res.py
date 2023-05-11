import json
from process_img import * 
from geotnf.transformation_high_res import GeometricTnf_high_res
from skimage import io
from glob import glob
from register_images import output_results
import torch
import gc

half_out_size=512

def get_data(filename): 
    with open(filename) as file:
        file_contents = file.read()
    
    json_data = json.loads(file_contents)
    
    return json_data
    

def transform_high_res(transforms, path, use_cuda=True):
    gc.collect()
    torch.cuda.empty_cache()
    
    # Get transforms
    theta_aff_1, theta_aff_2, theta_tps = transforms
        
    # Load geometric models
    tpsTnf = GeometricTnf_high_res(geometric_model='tps',    out_h=2*half_out_size, out_w=2*half_out_size, use_cuda=use_cuda)
    affTnf = GeometricTnf_high_res(geometric_model='affine', out_h=2*half_out_size, out_w=2*half_out_size, use_cuda=use_cuda)
    
    # Preprocess image 
    source_image     = io.imread(path)
    source_image_var = process_image(source_image, use_cuda, half_out_size=half_out_size, high_res=True)

    # Affine transformations
    warped_image_aff = affTnf(source_image_var, theta_aff_1.view(-1,2,3))
    warped_image_aff = affTnf(warped_image_aff, theta_aff_2.view(-1,2,3))
   
    # TPS transformation
    warped_image_aff_tps = tpsTnf(warped_image_aff,theta_tps)
     
    # Un-normalize images and convert to numpy
    warped_image_aff_tps_np = normalize_image(warped_image_aff_tps,forward=False).data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()
    
    # Ignore negative values
    warped_image_aff_tps_np[warped_image_aff_tps_np < 0] = 0    
   
    return warped_image_aff_tps_np

def create_tensor(transform, use_cuda=True):
    transform = torch.Tensor(transform)
    
    if use_cuda:
        transform = transform.cuda()
        
    return transform


samples = ["HMU_010_FH"]

def main():
    for sid in samples:
        json_path  = './transforms/transform_' + sid + '.json'
        img_path   = os.path.join('./results/preprocess/hist' , sid+'_high_res/')
        
        # Get image paths
        all_paths = {}
        all_paths['histo'] = sorted(glob(img_path + 'hist*.png'))
        all_paths['mask']  = sorted(glob(img_path + 'mask*.png'))
        
        json_data = get_data(json_path)
        
        # Get sid params
        origin      = json_data["origin"]
        direction   = json_data["direction"]
        spacing     = json_data["spacing"]
        
        scale       = json_data["scale"]
        hist_space  = [spacing[0]/scale[0], spacing[1]/scale[1], spacing[2]]
        
        spatialInfo = (origin, hist_space, direction)
        
        # Get image size
        img     = io.imread(all_paths['histo'][0])
        #w, h, _ = img.shape
        count   = len(all_paths['histo'])
        
        warped_imgs = {}
        for annot in all_paths:
            warped_imgs[annot] = np.zeros((count, 2*half_out_size, 2*half_out_size, 3))
        
        # Get transformation params
        for i in range(count):
            theta_aff_1 = json_data[str(i)]["affine_1"]
            theta_aff_2 = json_data[str(i)]["affine_2"]
            theta_tps   = json_data[str(i)]["tps"]
            transforms  = (create_tensor(theta_aff_1), create_tensor(theta_aff_2), create_tensor(theta_tps))
            
            for annot in all_paths:
                warped_imgs[annot][i,:,:,:] = transform_high_res(transforms, all_paths[annot][i], use_cuda=True)
            
            # Apply mask to image
            mask = np.zeros((2*half_out_size, 2*half_out_size, 3), dtype=int)
            for n in range(3):
                mask[:, :, n]  = warped_imgs['mask'][i, :, :, 0] 
            warped_imgs['histo'][i,:,:,:] = np.multiply(warped_imgs['histo'][i,:,:,:],mask)
            
        # Save images
        for annot in all_paths:
            output_results('./results/registration/', warped_imgs[annot], sid + '_high_res', annot, spatialInfo, extension = '.nii.gz')
        
main()
