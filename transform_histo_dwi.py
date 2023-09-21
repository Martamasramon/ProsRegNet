import json
from process_img                import * 
from transform_functions        import *
from glob                       import glob
from register_functions         import output_results
import cv2

samples = ['HMU_003_DB', 'HMU_007_TN', 'HMU_010_FH']

def main():
    for sid in samples:
        # Define resolution
        half_out_size = 512
        
        # Get info from json files
        json_path_histo_ex_vivo     = './transforms/transform_' + sid + '.json'
        json_path_ex_vivo_in_vivo   = './transforms/transform_' + sid + '_ex_vivo_in_vivo.json'
        json_data_histo_ex_vivo     = get_data(json_path_histo_ex_vivo)
        json_data_ex_vivo_in_vivo   = get_data(json_path_ex_vivo_in_vivo)
        
        # Get image paths
        img_path   = os.path.join('./results/preprocess/hist' , sid+'_high_res/')
        all_paths = {}
        all_paths['_histo'] = sorted(glob(img_path + 'hist*.png'))
        all_paths['_mask']  = sorted(glob(img_path + 'mask*.png'))
        count               = len(all_paths['_histo'])
        
        # Get coordinate information of dwi
        with open('coord.txt') as f:
            coord = json.load(f)
        slices = coord[sid]['slice']
        
        in_vivo_path    = os.path.join('./results/preprocess/mri/' , sid, 'mriUncropped_' + sid + '_' + str(slices[0]).zfill(2) + '.jpg')
        w, h, _         = (cv2.imread(in_vivo_path)).shape
        padding_factor  = int(round(max(np.add(coord[sid]['h'],np.multiply(2,coord[sid]['y_offset'])))/(coord[sid]['h'][0]+2*coord[sid]['y_offset'][0])))
        y_s             = (half_out_size*(2+2*padding_factor))/h
        x_s             = (half_out_size*(2+2*padding_factor))/w  
                    
        # Get sid params
        origin      = json_data_ex_vivo_in_vivo["origin"]        
        direction   = json_data_ex_vivo_in_vivo["direction"]    
        spacing     = json_data_ex_vivo_in_vivo["spacing"]       
        
        hist_space  = [spacing[0]/x_s, spacing[1]/y_s, spacing[2]]
        spatialInfo = (origin, hist_space, direction)
        
        warped_imgs = {}
        out_histo   = {}
        for annot in all_paths:
            warped_imgs[annot] = np.zeros((count, 2*half_out_size, 2*half_out_size, 3))
            out_histo[annot]   = np.zeros((count, half_out_size*(2+2*padding_factor), half_out_size*(2+2*padding_factor), 3))

        # Get transformation params
        for i in range(count):
            theta_aff_1 = json_data_histo_ex_vivo[str(i)]["affine_1"]
            theta_aff_2 = json_data_histo_ex_vivo[str(i)]["affine_2"]
            theta_tps_1 = json_data_histo_ex_vivo[str(i)]["tps"]
            
            theta_aff_3 = json_data_ex_vivo_in_vivo[str(i)]["affine_1"]
            theta_aff_4 = json_data_ex_vivo_in_vivo[str(i)]["affine_2"]
            theta_tps_2 = json_data_ex_vivo_in_vivo[str(i)]["tps"]
            transforms  = (create_tensor(theta_aff_1), create_tensor(theta_aff_2), create_tensor(theta_tps_1), 
                            create_tensor(theta_aff_3), create_tensor(theta_aff_4), create_tensor(theta_tps_2))
            
            for annot in all_paths:
                warped_imgs[annot][i,:,:,:] = transform_histo_double(transforms, all_paths[annot][i], use_cuda=True, out_size=2*half_out_size)
            
            # Transform to MRI space
            new_size, start_x, end_x, start_y, end_y = resize_from_coord(coord, sid, i, x_s, y_s)
                        
            for annot in all_paths:
                out_histo[annot][i, start_x:end_x, start_y:end_y, :] = cv2.resize(warped_imgs[annot][i,:,:,:], new_size, interpolation=cv2.INTER_CUBIC)  

        # Save images
        for annot in all_paths:
            output_results('./results/registration/histo-ex-in-vivo/', out_histo[annot], sid, annot, spatialInfo, 'final', extension = '.nii.gz')
    
main()
