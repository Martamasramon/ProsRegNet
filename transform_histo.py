import json
from process_img            import * 
from transform_functions    import *
from glob                   import glob
from register_functions     import output_results
import argparse
import cv2

samples      = ['HMU_011_MQ','HMU_056_JH','HMU_077_MW','HMU_087_FM','HMU_180_KF','HMU_242_JD','HMU_348_EA']

# 'HMU_010_FH',  'HMU_025_SH', 'HMU_038_JC', 'HMU_063_RS', 'HMU_066_JF', 
#     'HMU_069_NS', 'HMU_076_RV', 'HMU_082_PS', 'HMU_084_AJ','HMU_113_MT', 'HMU_121_CN', 
#     'HMU_176_IJ', 'HMU_180_KF', 'HMU_201_MB'

"""['HMU_003_DB','HMU_004_HC','HMU_007_TN','HMU_011_MQ','HMU_033_JS','HMU_056_JH','HMU_060_CH', 'HMU_064_SB', 
                  'HMU_065_RH','HMU_067_MS','HMU_068_PB','HMU_077_MW','HMU_087_FM','HMU_094_RB', 'HMU_099_DL', 
                  'HMU_116_BC','HMU_118_PL', 'HMU_119_MM','HMU_128_RK','HMU_174_IS', 'HMU_181_MO','HMU_198_JL', 
                  'HMU_226_NS','HMU_227_KT','HMU_235_CC', 'HMU_242_JD', 'HMU_245_DC', 'HMU_256_DB','HMU_258_JK',
                  'HMU_265_JM', 'HMU_342_ME','HMU_348_EA','HMU_258_JK']"""

save_png_img = False

def main():
    
    ###### INPUTS
    parser = argparse.ArgumentParser(description='Parse data')
    parser.add_argument('-m',   '--mri',  type=str, default='T2', help='mri type')

    opt     = parser.parse_args()
    flip_v  = False
    flip_h  = False
    reverse = False
    if opt.mri == 'T2':
        folder      = 'histo-T2/'
        coord_path  = 'coord.txt'
        mri_folder  = 'mri/'
        reverse     = True
    elif opt.mri == 'b0':
        folder      = 'histo-b0/'
        coord_path  = 'coord_dwi_b0.txt'
        mri_folder  = 'dwi-b0/'
        flip_h      = False ### ???
    elif opt.mri == 'b90':
        folder      = 'histo-b90/'
        coord_path  = 'coord_dwi_b90.txt'
        mri_folder  = 'dwi-b90/'
    elif opt.mri == 'fIC':
        folder      = 'histo-fIC/'
        coord_path  = 'coord_dwi_fIC.txt'
        mri_folder  = 'fIC/'
    else:
        print('ERROR. Input modality not recognised.')
        return
    
    for sid in samples:
        print('Processing ' + sid + '...')
        # Define resolution
        half_out_size = 512
        
        # Get info from json files
        json_path  = './transforms/' + folder + 'transform_' + sid + '.json'
        json_data  = get_data(json_path)
        
        # Get image paths
        img_path   = os.path.join('./results/preprocess/hist' , sid+'_high_res/')
        all_paths = {}
        all_paths['histo']      = sorted(glob(img_path + 'hist*.png'),          reverse=reverse)
        all_paths['mask']       = sorted(glob(img_path + 'mask*.png'),          reverse=reverse)
        all_paths['density']    = sorted(glob(img_path + 'density*.png'),       reverse=reverse)
        all_paths['stroma']     = sorted(glob(img_path + 'stroma*.png'),        reverse=reverse)
        all_paths['lumen']      = sorted(glob(img_path + 'lumen*.png'),         reverse=reverse)
        all_paths['epithelial'] = sorted(glob(img_path + 'epithelial*.png'),    reverse=reverse)
        all_paths['cancer']     = sorted(glob(img_path + 'cancer*.png'),        reverse=reverse)
        all_paths['BPH']        = sorted(glob(img_path + 'BPH*.png'),           reverse=reverse)
        all_paths['urethra']    = sorted(glob(img_path + 'urethra*.png'),       reverse=reverse)
        
        paths = dict(all_paths)
        for annot in all_paths:
            if len(all_paths[annot])==0:
                del paths[annot]
                
        count = len(paths['histo'])
        
        # Get coordinate information of MRI
        with open(coord_path) as f:
            coord = json.load(f)
        slices = coord[sid]['slice']
        
        mri_path        = os.path.join('./results/preprocess/' + mri_folder, sid, 'mri_uncropped_' + sid + '_' + str(slices[0]).zfill(2) + '.jpg')
        w, h, _         = (cv2.imread(mri_path)).shape
        padding_factor  = int(round(max(np.add(coord[sid]['h'],np.multiply(2,coord[sid]['y_offset'])))/(coord[sid]['h'][0]+2*coord[sid]['y_offset'][0])))
        y_s             = (half_out_size*(2+2*padding_factor))/h
        x_s             = (half_out_size*(2+2*padding_factor))/w  
        #scale = 2*half_out_size/80
                    
        # Get sid params
        origin      = json_data["origin"]        
        direction   = json_data["direction"]    
        spacing     = json_data["spacing"]   
        
        hist_space  = [spacing[0]/x_s, spacing[1]/y_s, spacing[2]]
        spatialInfo = (origin, hist_space, direction)
        
        warped_imgs = {}
        out_histo   = {}
        for annot in paths:
            warped_imgs[annot] = np.zeros((count, 2*half_out_size, 2*half_out_size, 3))
            out_histo[annot]   = np.zeros((count, half_out_size*(2+2*padding_factor), half_out_size*(2+2*padding_factor), 3))

        # Get transformation params
        for i in range(count):
            theta_aff_1 = json_data[str(i)]["affine_1"]
            theta_aff_2 = json_data[str(i)]["affine_2"]
            theta_tps   = json_data[str(i)]["tps"]
            
            transforms  = (create_tensor(theta_aff_1), create_tensor(theta_aff_2), create_tensor(theta_tps))
            
            for annot in paths:
                warped_imgs[annot][i,:,:,:] = transform_histo(transforms, paths[annot][i], flip_v, flip_h, use_cuda=True, out_size=2*half_out_size)
            
            if save_png_img:
                # Save png image
                cv2.imwrite('./results/registration/' + folder + sid + '/' + sid + '_' + str(i) +'.png', warped_imgs['histo'][i,:,:,:]*255)
            
            # Transform to MRI space
            new_size, start_x, end_x, start_y, end_y = resize_from_coord(coord, sid, i, x_s, y_s)
                        
            for annot in paths:
                out_histo[annot][i, start_x:end_x, start_y:end_y, :] = cv2.resize(warped_imgs[annot][i,:,:,:], new_size, interpolation=cv2.INTER_CUBIC)  
                if flip_v:
                    out_histo[annot][i,:,:,:] = cv2.flip(out_histo[annot][i,:,:,:] , 0)
            
        # Save images
        for annot in paths:
            print(annot)
            output_results('./results/registration/' + folder, out_histo[annot], sid, '_final_' + annot, spatialInfo, 'final', extension = '.nii.gz')

        print('Finished processing sample. \n')
        
    print('Done!')
main()
