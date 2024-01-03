from __future__ import print_function, division
import os
import json
import cv2
import torch
import json
import SimpleITK    as sitk
import numpy        as np

from image.normalization        import normalize_image
from geotnf.transformation      import GeometricTnf
from geotnf.point_tnf           import *
from process_img                import *
from skimage                    import io
from preprocess                 import *
from register_functions         import *
from landmark_functions         import *
from model.ProsRegNet_model     import ProsRegNet
from collections                import OrderedDict
from scipy.spatial.distance     import directed_hausdorff


import warnings
warnings.filterwarnings('ignore')

    
def save_transform(theta_aff_1,theta_aff_2,theta_tps):
    
    json_data = {
        "affine_1": theta_aff_1.cpu().detach().numpy().tolist(),
        "affine_2": theta_aff_2.cpu().detach().numpy().tolist(),
        "tps":      theta_tps.cpu().detach().numpy().tolist()
    }
    
    return json_data


def save_all_transforms(json_data, file_name, imSpatialInfo, scaling, folder):
    # Add spatial information to transformation data
    json_data['origin']     = imSpatialInfo[0]
    json_data['spacing']    = imSpatialInfo[1]
    json_data['direction']  = imSpatialInfo[2]
    json_data['scale']      = scaling
    
    json_object = json.dumps(json_data, indent=4)
    
    # Save to file
    with open('./transforms/' + folder + 'transform_' + file_name + '.json', "w") as outfile:
        outfile.write(json_object)


def load_models(model_aff_path, model_tps_path, do_deformable=True, tps_type='tps', mri=False, feature_extraction_cnn = 'resnet101'): 
    """ 
    Load pre-trained models
    """
    
    use_cuda = torch.cuda.is_available()
    
    do_aff = not model_aff_path==''
    do_tps = do_deformable
    
    # Create model
    print('Creating CNN model...')    
    if do_aff:
        model_aff = ProsRegNet(use_cuda=use_cuda,geometric_model='affine',feature_extraction_cnn=feature_extraction_cnn, mri=mri)
    if do_tps:
        model_tps = ProsRegNet(use_cuda=use_cuda,geometric_model=tps_type,feature_extraction_cnn=feature_extraction_cnn, mri=mri)

    # Load trained weights
    print('Loading trained model weights...')
    if do_aff:
        checkpoint               = torch.load(model_aff_path, map_location=lambda storage, loc: storage)
        checkpoint['state_dict'] = OrderedDict([(k.replace('resnet101', 'model'), v) for k, v in checkpoint['state_dict'].items()])
        model_aff.load_state_dict(checkpoint['state_dict'])
    if do_tps:
        checkpoint               = torch.load(model_tps_path, map_location=lambda storage, loc: storage)
        checkpoint['state_dict'] = OrderedDict([(k.replace('resnet101', 'model'), v) for k, v in checkpoint['state_dict'].items()])
        model_tps.load_state_dict(checkpoint['state_dict'])
    
    model_cache = (model_aff, model_tps, do_aff, do_tps, use_cuda)
    
    return model_cache


def output_results(outputPath, inputStack, sid, fn, imSpatialInfo, model, extension = "nii.gz"):
    """
    Output results to .nii.gz volumes
    """
    
    if len(inputStack.shape)<4:
        c, w, h     = inputStack.shape
        temp        = inputStack
        inputStack  = np.zeros((c,w,h,3))
        for i in range(3):
            inputStack[:,:,:,i] = temp
        
    mriOrigin, mriSpace, mriDirection = imSpatialInfo
    sitkIm = sitk.GetImageFromArray(inputStack)
    
    sitkIm.SetOrigin(mriOrigin)
    sitkIm.SetSpacing(mriSpace)
    sitkIm.SetDirection(mriDirection)
    try: 
        os.mkdir(outputPath + sid)
    except: 
        pass
    
    #sitk.WriteImage(sitkIm, outputPath + sid + '/' + model + '_' + sid + fn + extension)
    sitk.WriteImage(sitkIm, outputPath + sid + '/'+ sid + fn + extension)


def getFiles(file_dest, keyword, sid, reverse=False): 
    cases = []
    files = [pos for pos in sorted(os.listdir(file_dest),reverse=reverse) if keyword in pos]

    for f in files: 
        if sid in f: 
            cases.append(f)
            
    return cases
    

def calc_dice(histo_mask, mri_mask):
    """
    calculate DICE coefficient between masks
    """
    
    histo_mask[histo_mask > 0.5]  = 1
    histo_mask[histo_mask <= 0.5] = 0   
    
    count, h, w = histo_mask.shape  
    dice_total  = 0
    
    if count > 1:
        print('---- DICE coefficient ----')
    
    if np.max(mri_mask) > 1:
        mri_mask /= 255
        
    for i in range(count):        
        if histo_mask.shape != mri_mask.shape:
            # Resize so same dimensions
            mri = cv2.resize(mri_mask[i,:,:], (w, h), interpolation=cv2.INTER_CUBIC)
            mri[mri > 0.5]  = 1
            mri[mri <= 0.5] = 0 
        else:
            mri = mri_mask[i,:,:]
            
        histo = histo_mask[i,:,:]
        
        # Calculate DICE
        try:            
            numerator   = 2 * np.sum(np.multiply(histo,mri))
            denominator = np.sum(histo + mri)
            dice        = np.sum(numerator/(denominator + 0.00001)) 
            
            if count > 1:
                print('Slice ' + str(i) + ': ' + str(dice))
        except:
            try:
                numerator   = 2 * torch.sum(torch.multiply(histo,mri))
                denominator = torch.sum(histo + mri)
                dice        = torch.sum(numerator/(denominator + 0.00001)) 
                dice        = dice.data.cpu().numpy()
                
                if count > 1:
                    print('Slice ' + str(i) + ': ' + str(dice))
            except:
                dice = 1
                print('Error calculating DICE.')

        dice_total += dice/count
        
    print('Average DICE: ' + str(dice_total))
    return dice_total

def hausdorff(mask_A, mask_B):
    """
    calculate DICE coefficient between masks
    """ 
          
    mask_A[mask_A > 0.5]  = 1
    mask_A[mask_A <= 0.5] = 0
        
    count, h, w = mask_A.shape
    total       = 0
    
    if count > 1:
        print('---- Hausdorff distance ----')
        
    if np.max(mask_B) > 1:
        mask_B /= 255
        
    for i in range(count):
        if mask_A.shape != mask_B.shape:
            # Resize so same dimensions
            B = cv2.resize(mask_B[i,:,:], (w, h), interpolation=cv2.INTER_CUBIC)
            B[B > 0.5]  = 1
            B[B <= 0.5] = 0 
        else:
            B = mask_B[i,:,:]
        
        A = mask_A[i,:,:]
        
        # Calculate Hausdorff distance
        try:
            x_A, y_A  = np.nonzero(A)    
            x_B, y_B  = np.nonzero(B)
            
            non_zero_A = np.zeros((len(x_A), 2))
            non_zero_A[:,0] = x_A
            non_zero_A[:,1] = y_A
            
            non_zero_B = np.zeros((len(x_B), 2))
            non_zero_B[:,0] = x_B
            non_zero_B[:,1] = y_B
            
            hausdorff_A = directed_hausdorff(non_zero_A, non_zero_B)  
            hausdorff_B = directed_hausdorff(non_zero_B, non_zero_A)  
            
            hausdorff   = (hausdorff_A[0] + hausdorff_B[0])/2
            
            if count > 1:
                print('Slice ' + str(i) + ': ', hausdorff)
  
        except:
            hausdorff = 1
            print('Error calculating Hausdorff distance.')

        total += hausdorff/count
        
    print('Average Hausdorff distance: ' + str(total))
    return total

def runCnn(model_cache, source_image_path, target_image_path, histo_regions, out_size=240, mri=False, exvivo=False):
    """
    Run the cnn on images and return 3D images
    """
    
    
    model_aff, model_tps, do_aff, do_tps, use_cuda = model_cache
    
    affTnf = GeometricTnf(geometric_model='affine', out_h=out_size, out_w=out_size, use_cuda=use_cuda)
    if mri and not exvivo:
        tpsTnf = GeometricTnf(geometric_model='tps-mri', out_h=out_size, out_w=out_size, use_cuda=use_cuda)
    else:
        tpsTnf = GeometricTnf(geometric_model='tps', out_h=out_size, out_w=out_size, use_cuda=use_cuda)
    
    source_image = io.imread(source_image_path)
    target_image = io.imread(target_image_path)
    
    if mri:
        # copy MRI image to 3 channels 
        source_image3d = np.zeros((source_image.shape[0], source_image.shape[1], 3), dtype=int)
        source_image3d[:, :, 0] = source_image
        source_image3d[:, :, 1] = source_image
        source_image3d[:, :, 2] = source_image
        source_image = np.copy(source_image3d)

    # copy MRI image to 3 channels 
    target_image3d = np.zeros((target_image.shape[0], target_image.shape[1], 3), dtype=int)
    target_image3d[:, :, 0] = target_image
    target_image3d[:, :, 1] = target_image
    target_image3d[:, :, 2] = target_image
    target_image = np.copy(target_image3d)


    ##### Preprocess #####
    # Masks 
    source_image_mask_var = process_image(source_image, use_cuda, out_size=out_size, mask=True)
    target_image_mask_var = process_image(target_image, use_cuda, out_size=out_size, mask=True)
    
    # Images 
    source_image_var = process_image(source_image, use_cuda, out_size=out_size)
    target_image_var = process_image(source_image, use_cuda, out_size=out_size)
    
    histo_image_var  = {}
    for region in histo_regions:
        if region == 'density':
            histo_image_var[region]  = process_image(histo_regions[region], use_cuda, out_size=out_size)
        else:
            histo_image_var[region]  = process_image(histo_regions[region], use_cuda, out_size=out_size, mask=True)

    ##### Evaluate models #####
    if do_aff:
        model_aff.eval()
    if do_tps:
        model_tps.eval()

    # Registration inputs
    batch_mask      = {'source_image': source_image_mask_var,       'target_image':target_image_mask_var}
    batch           = {'source_image': source_image_var,            'target_image':target_image_var}
    
    if do_aff:
        ######## Find affine registration using masks only -> TWICE
        ## 1. 
        theta_aff_1     = model_aff(batch_mask)
        warped_mask_aff = affTnf(batch_mask['source_image'], theta_aff_1.view(-1,2,3))     
        ## 2. 
        second_batch_mask = {'source_image': warped_mask_aff, 'target_image':target_image_mask_var}
        theta_aff_2       = model_aff(second_batch_mask)
        
        # Transform images
        warped_image_aff            = affTnf(batch['source_image'], theta_aff_1.view(-1,2,3))
        warped_image_aff            = affTnf(warped_image_aff, theta_aff_2.view(-1,2,3))
        
        warped_aff  = {}
        for region in histo_regions:
            warped_aff[region]          = affTnf(histo_image_var[region], theta_aff_1.view(-1,2,3))
            warped_aff[region]          = affTnf(warped_aff[region], theta_aff_2.view(-1,2,3))
            
    
    if do_aff and do_tps:
        ######## TPS registration - low res images 
        theta_tps               = model_tps({'source_image': warped_image_aff, 'target_image': batch['target_image']})   
        warped_image_aff_tps    = tpsTnf(warped_image_aff,theta_tps)

        warped_aff_tps  = {}
        for region in histo_regions:
            warped_aff_tps[region] = tpsTnf(warped_aff[region], theta_tps)
                   
    transforms_json = save_transform(theta_aff_1,theta_aff_2,theta_tps)
    transforms      = (theta_aff_1,theta_aff_2,theta_tps)

    ### Un-normalize images and convert to numpy
    if do_aff and do_tps:
        warped_image_np = normalize_image(warped_image_aff_tps,forward=False).data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()

        warped_regions_np  = {}
        for region in histo_regions:
            warped_regions_np[region] = normalize_image(warped_aff_tps[region],forward=False).data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()
            
    elif do_aff:
        warped_image_np = normalize_image(warped_image_aff,forward=False).data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()

        warped_regions_np  = {}
        for region in histo_regions:
            warped_regions_np[region] = normalize_image(warped_aff[region],forward=False).data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy() 

    # Ignore negative values
    warped_image_np[warped_image_np < 0]         = 0    
   
    return warped_image_np, warped_regions_np, transforms_json, transforms

def get_map(preprocess_fixed_dest, text='fIC'):
    
    files       = [pos for pos in sorted(os.listdir(preprocess_fixed_dest)) if text in pos]
    w, h, _     = (cv2.imread(preprocess_fixed_dest + files[0])).shape 
    count       = len(files)
    out3D       = np.zeros((count, w, h, 3))
    
    for idx in range(count): 
        img               = cv2.imread(preprocess_fixed_dest  + files[idx])
        out3D[idx,:,:,:]  = np.uint8(img)

    return out3D

   
def register(preprocess_moving_dest, preprocess_fixed_dest, coord, model_cache, sid, regions, landmarks_histo=None, landmarks_mri=None, landmarks_grid=None, half_out_size = 120, mri=False, exvivo=False, fIC=None, reg_fIC=False):     
    if landmarks_histo and landmarks_mri:
        landmarks = True
    else:
        landmarks = False
        
    ### Grab MRI files that were preprocessed    
    mri_files = [pos_mri for pos_mri in sorted(os.listdir(preprocess_fixed_dest)) if pos_mri.endswith('.jpg') ]
    
    mri_case    = []
    mri_highRes = []
    mri_mask    = []
    mri_cancer  = []
    
    # Classify MRI images
    for mri_file in mri_files: 
        if sid in mri_file: 
            if 'uncropped' in mri_file: 
                mri_highRes.append(mri_file)
            elif 'mask' in mri_file: 
                mri_mask.append(mri_file)
            elif 'cancer' in (mri_file):
                mri_cancer.append(mri_file)
            elif 'mri' in (mri_file):  
                mri_case.append(mri_file)
           
                
    ### Grab histology files that were preprocessed     
    if fIC or mri or reg_fIC:
        reverse = False
    else:
        reverse = True
    if mri:
        hist_case  = getFiles(preprocess_moving_dest, 'mri_'+sid, sid, reverse=reverse)
    else:
        hist_case  = getFiles(preprocess_moving_dest, 'hist', sid, reverse=reverse)
    
    cases = {}
    for region in regions:
        cases[region] = getFiles(preprocess_moving_dest, region, sid, reverse=reverse)

    # Print out regions in histology
    print('\nRegions in registration:')
    for region in regions:
        print(region)

    ### Create empty arrays to store warped images
    # Find dimensions
    w, h, _         = (cv2.imread(preprocess_fixed_dest + mri_highRes[0])).shape
    count           = min(len(hist_case), len(mri_case))
    padding_factor  = int(round(max(np.add(coord[sid]['h'],np.multiply(2,coord[sid]['y_offset'])))/(coord[sid]['h'][0]+2*coord[sid]['y_offset'][0])))

    y_s = (half_out_size*(2+2*padding_factor))/h
    x_s = (half_out_size*(2+2*padding_factor))/w  
    
    array_size = half_out_size*(2+2*padding_factor)

    # Create arrays
    out3Dhist       = np.zeros((count, array_size, array_size, 3))
    out3Dmri        = np.zeros((count, w, h, 3))
    out3Dmri_mask   = np.zeros((count, w, h, 3)[:-1])
    if len(mri_cancer)>0:
        out3Dmri_cancer = np.zeros((count, w, h, 3))
    
    out3D_regions = {}
    for region in regions:
        out3D_regions[region]       = np.zeros((count, array_size, array_size))
        
    if fIC:
        out3D_regions['fIC']        = np.zeros((count, array_size, array_size, 3))
        for region in regions:
            if region != 'density':
                out3D_regions['fIC_'+region] = np.zeros((count, array_size, array_size))
            else:
                out3D_regions['fIC_'+region] = np.zeros((count, array_size, array_size, 3))
        
    if landmarks_mri:
        landmark_list   = ([pos for pos in sorted(os.listdir(preprocess_fixed_dest + 'landmarks/'))])
        landmark_image  = np.zeros((count, w, h, 3))
        
        landmark_cropped  = {}
        landmark_highRes  = {}
        for landmark_num in landmark_list:
            landmark_cropped[landmark_num]  = ([pos for pos in sorted(os.listdir(preprocess_fixed_dest + 'landmarks/' + landmark_num)) if 'cropped' in pos])
            landmark_highRes[landmark_num]  = ([pos for pos in sorted(os.listdir(preprocess_fixed_dest + 'landmarks/' + landmark_num)) if 'highRes' in pos])
    
    ## Transform each slice
    all_transforms     = {}    
    for idx in range(count): 
        # Get image paths
        source_image_path = preprocess_moving_dest + hist_case[idx]
        target_image_path = preprocess_fixed_dest  + mri_case[idx]
        
        # Read data from coord.txt
        x_prime         = coord[sid]['x'][idx]
        y_prime         = coord[sid]['y'][idx]
        x_offset_prime  = coord[sid]['x_offset'][idx]
        y_offset_prime  = coord[sid]['y_offset'][idx]
        h_prime         = coord[sid]['h'][idx]
        w_prime         = coord[sid]['w'][idx]
        
        # Calculate new dimensions
        w_new   = int(w_prime * x_s) + 2*int(x_offset_prime * x_s)
        h_new   = int(h_prime * y_s) + 2*int(y_offset_prime * y_s)
        start_x = int(x_prime * x_s) - int(x_offset_prime * x_s)
        end_x   = int(x_prime * x_s) + int(w_prime * x_s) + int(x_offset_prime * x_s)
        start_y = int(y_prime * y_s) - int(y_offset_prime * y_s)
        end_y   = int(y_prime * y_s) + int(h_prime * y_s) + int(y_offset_prime * y_s)
        
        # Get histology regions
        imHisto         = {}
        for region in regions:
            imHisto[region] = cv2.imread(preprocess_moving_dest + cases[region][idx])
        
        ######## REGISTER ########
        affTps, regions_aff_tps, transforms_json, transforms = runCnn(model_cache, source_image_path, target_image_path, imHisto, out_size=2*half_out_size, mri=mri, exvivo=exvivo) 
        all_transforms[idx] = transforms_json
        
        # Save transformed source image as png
        cv2.imwrite(preprocess_moving_dest + '/warped_' + sid + '_' + str(idx).zfill(2) +'.jpg', affTps*255)
        
        # Transform source image & regions to MRI space    
        affTps = cv2.resize(affTps*255, (h_new, w_new), interpolation=cv2.INTER_CUBIC)  
        for region in regions:
            regions_aff_tps[region] = cv2.resize(regions_aff_tps[region], (h_new, w_new), interpolation=cv2.INTER_CUBIC)
            regions_aff_tps[region] = regions_aff_tps[region]*255
            
            if region != 'density':
                regions_aff_tps[region] = regions_aff_tps[region]>255/1.5
            else:
                regions_aff_tps[region][regions_aff_tps[region]<0] = 0
            
        ## Create mask for warped histology
        mask_image3d  = np.zeros((affTps.shape[0], affTps.shape[1], 3), dtype=int)
        for i in range(3):
            mask_image3d[:, :, i]  = regions_aff_tps['mask'][:, :, 0]
        affTps = np.multiply(affTps,mask_image3d)   
            
        # Output histology & regions
        out3Dhist[idx, start_x:end_x, start_y:end_y, :] = np.uint8(affTps)
        for region in regions:
            out3D_regions[region][idx, start_x:end_x, start_y:end_y] = np.uint8(regions_aff_tps[region][:, :, 0])
             
        """
        # Not necessary for MLP fIC maps
        if fIC:            
            out3D_regions['fIC'][idx, start_x:end_x, start_y:end_y, :]  = np.uint8(affTps)
            out3D_regions['fIC'][idx, :,:,:] = cv2.flip(out3D_regions['fIC'][idx, :,:,:],0)
            for region in regions:
                if region != 'density':
                    out3D_regions['fIC_'+region ][idx, start_x:end_x, start_y:end_y]  = np.uint8(regions_aff_tps[region][:, :, 0])
                    out3D_regions['fIC_'+region ][idx, :,:] = cv2.flip(out3D_regions['fIC_'+region ][idx, :,:],0)
                else:
                    out3D_regions['fIC_'+region][idx, start_x:end_x, start_y:end_y, :] = np.uint8(regions_aff_tps[region])
                    out3D_regions['fIC_'+region][idx, :,:,:] = cv2.flip(out3D_regions['fIC_'+region][idx, :,:,:],0)"""
            
        # Transform & output histology landmarks
        if landmarks_grid:
            if mri:
                file_name = 'landmarks_mri'
                suffix    = 'mri_'
            else:
                file_name = 'landmarks'
                suffix    = ''
                
            with open('jsonData/'+ file_name +'.json') as f:
                landmarks_grid = json.load(f)
                x = np.array(landmarks_grid['x'].split(';')).astype(float)
                y = np.array(landmarks_grid['y'].split(';')).astype(float)
                landmarks_grid_img = list_to_image((half_out_size*2,half_out_size*2), x, y)
                
            landmark_loc_histo = transform_landmarks(landmarks_grid_img, transforms, mri=mri, out_size=half_out_size*2)
            
            ## SAVE AS TEXT 
            with open('./results/landmarks/transformed_landmarks_' + suffix + sid + '_' + str(idx) + '.txt', 'w') as file:  
                x_transformed = landmark_loc_histo[0].cpu().numpy().astype(str)
                y_transformed = landmark_loc_histo[1].cpu().numpy().astype(str)
                x_str = ";".join(x_transformed)
                y_str = ";".join(y_transformed)
                file.write(f'{x_str}')
                file.write(f'\n{y_str}')
            
            # landmarks_histo is size 240x240xnum_landmarks
                
        # Output MRI
        imMri_highRes   = cv2.imread(preprocess_fixed_dest  + mri_highRes[idx])
        imMriMask       = cv2.imread(preprocess_fixed_dest  + mri_mask[idx])
        
        out3Dmri[idx,:,:,:]         = np.uint8(imMri_highRes)
        out3Dmri_mask[idx, :, :]    = np.uint8((imMriMask[:, :, 0] > 255/2.0))
        
        try:
            imMriCancer = cv2.imread(preprocess_fixed_dest  + mri_cancer[idx])
            out3Dmri_cancer[idx, :, :, :] = np.uint8(imMriCancer)
        except:
            print('error with cancer mask')
            out3Dmri_cancer = np.zeros(())
        
        if landmarks:  
            # To save as NIFTI
            #landmark_histo_img = list_to_image((half_out_size*2,half_out_size*2), landmark_loc_histo[0,:], landmark_loc_histo[1,:], 2, compact=True)
            #landmark_histo_img = cv2.resize(landmark_histo_img, (h_new, w_new), interpolation=cv2.INTER_CUBIC)  
            
            #all_landmarks_histo = np.zeros((count, array_size, array_size,3 ))
            #all_landmarks_histo[idx,start_x:end_x, start_y:end_y,:] = np.uint8(landmark_histo_img)
            
            # Histo for error calc
            #temp_all      = np.zeros((array_size, array_size, landmark_loc_histo.shape[1]))
            #landmark_temp = list_to_image((half_out_size*2,half_out_size*2), landmark_loc_histo[0,:], landmark_loc_histo[1,:], 2)   
            #landmark_temp = resize_landmarks(landmark_temp, h_new, w_new) 
            #temp_all[start_x:end_x, start_y:end_y,:] = np.uint8(landmark_temp)
            #landmark_loc_histo = image_to_list(temp_all,  hist=False)
                      
            ## Use preprocessed imgs to calculate error
            all_landmarks   = np.zeros((half_out_size*4,half_out_size*4,len(landmark_list)))
            
            for landmark_num in landmark_list:
                img_highRes  = cv2.imread(preprocess_fixed_dest + 'landmarks/' + landmark_num + '/' + landmark_highRes[landmark_num][idx])
                # For saving
                landmark_image[:,:,:,0] += img_highRes[:,:,0]
                # For error calc
                img_landmark = cv2.resize(img_highRes, (half_out_size*4, half_out_size*4), interpolation=cv2.INTER_CUBIC)
                all_landmarks[:,:,int(landmark_num)] = img_landmark[:,:,0]
            
            landmark_loc_mri    = image_to_list(all_landmarks, hist=False)    
            landmark_loc_histo  = landmark_loc_histo.cpu().detach()
            
            landmark_MSE = torch.mean((landmark_loc_mri - landmark_loc_histo)*(landmark_loc_mri - landmark_loc_histo))
            print('MSE: ', landmark_MSE.numpy())
            print('SE per point: ', torch.mean((landmark_loc_mri - landmark_loc_histo)*(landmark_loc_mri - landmark_loc_histo),0).numpy())
        else:
            all_landmarks_histo = {}
            landmark_image = {}
            landmark_MSE   = 0
            
    output3D_cache = (out3Dhist, out3Dmri, out3Dmri_cancer, out3D_regions, out3Dmri_mask, [x_s,y_s], all_transforms, landmark_image, all_landmarks_histo)
    
    return output3D_cache
    