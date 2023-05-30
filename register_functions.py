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
from model.ProsRegNet_model     import ProsRegNet
from collections                import OrderedDict

import warnings
warnings.filterwarnings('ignore')

    
def save_transform(theta_aff_1,theta_aff_2,theta_tps):
    
    json_data = {
        "affine_1": theta_aff_1.cpu().detach().numpy().tolist(),
        "affine_2": theta_aff_2.cpu().detach().numpy().tolist(),
        "tps":      theta_tps.cpu().detach().numpy().tolist()
    }
    
    return json_data


def save_all_transforms(json_data, file_name, imSpatialInfo, scaling):
    # Add spatial information to transformation data
    json_data['origin']     = imSpatialInfo[0]
    json_data['spacing']    = imSpatialInfo[1]
    json_data['direction']  = imSpatialInfo[2]
    json_data['scale']      = scaling
    
    json_object = json.dumps(json_data, indent=4)
    
    # Save to file
    with open('./transforms/transform_' + file_name + '.json', "w") as outfile:
        outfile.write(json_object)


def load_models(feature_extraction_cnn, model_aff_path, model_tps_path, do_deformable=True, tps_type='tps'): 
    """ 
    Load pre-trained models
    """
    
    use_cuda = torch.cuda.is_available()
    
    do_aff = not model_aff_path==''
    do_tps = do_deformable

    # Create model
    print('Creating CNN model...')
    if do_aff:
        model_aff = ProsRegNet(use_cuda=use_cuda,geometric_model='affine',feature_extraction_cnn=feature_extraction_cnn)
    if do_tps:
        model_tps = ProsRegNet(use_cuda=use_cuda,geometric_model=tps_type,feature_extraction_cnn=feature_extraction_cnn)

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
    
    mriOrigin, mriSpace, mriDirection = imSpatialInfo
    sitkIm = sitk.GetImageFromArray(inputStack)
    
    sitkIm.SetOrigin(mriOrigin)
    sitkIm.SetSpacing(mriSpace)
    sitkIm.SetDirection(mriDirection)
    try: 
        os.mkdir(outputPath + sid)
    except: 
        pass
    sitk.WriteImage(sitkIm, outputPath + sid + '/' + model + '_' + sid + fn + extension)


def getFiles(file_dest, keyword, sid): 
    cases = []
    files = [pos for pos in sorted(os.listdir(file_dest)) if keyword in pos]

    for f in files: 
        if sid in f: 
            cases.append(f)
            
    return cases
    

def calc_dice(hist_mask, mri_mask):
    """
    calculate DICE coefficient between masks
    """
    count, h, w = hist_mask.shape
    dice_total  = 0
    
    print('---- DICE coefficient ----')
    for i in range(count):
        # Resize so same dimensions
        mri   = cv2.resize(mri_mask[i,:,:], (w, h), interpolation=cv2.INTER_CUBIC)
        histo = hist_mask[i,:,:]
        
        # Calculate DICE
        try:
            numerator   = 2 * np.sum(np.multiply(histo,mri))
            denominator = np.sum(histo + mri)
            dice        = np.sum(numerator/(denominator + 0.00001)) 
            print('Slice ' + str(i) + ': ' + str(dice))
        except:
            dice = 1
            print('Error calculating DICE.')
        
        dice_total += dice/count
        
    print('Total DICE: ' + str(dice_total))


def runCnn(model_cache, source_image_path, target_image_path, histo_regions, mri=False):
    """
    Run the cnn on images and return 3D images
    """
    
    
    model_aff, model_tps, do_aff, do_tps, use_cuda = model_cache
    
    if mri:
        tpsTnf = GeometricTnf(geometric_model='tps-mri',    use_cuda=use_cuda)
    else:
        tpsTnf = GeometricTnf(geometric_model='tps',    use_cuda=use_cuda)
    affTnf = GeometricTnf(geometric_model='affine', use_cuda=use_cuda)
    
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


    ##### Preprocess masks - low res
    source_image_mask_var = process_image(source_image, use_cuda, mask=True)
    target_image_mask_var = process_image(target_image, use_cuda, mask=True)

    #### Preprocess images 
    source_image_var = process_image(source_image, use_cuda)
    target_image_var = process_image(source_image, use_cuda)
    
    histo_image_var  = {}
    for region in histo_regions:
        histo_image_var[region]          = process_image(histo_regions[region], use_cuda)

    ##### Evaluate models
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
        theta_aff_tps           = model_tps({'source_image': warped_image_aff, 'target_image': batch['target_image']})   
        warped_image_aff_tps    = tpsTnf(warped_image_aff,theta_aff_tps)

        warped_aff_tps  = {}
        for region in histo_regions:
            warped_aff_tps[region] = tpsTnf(warped_aff[region], theta_aff_tps)
                   
    transform = save_transform(theta_aff_1,theta_aff_2,theta_aff_tps)

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
   
    return warped_image_np, warped_regions_np, transform


   
def register(preprocess_moving_dest, preprocess_fixed_dest, coord, model_cache, sid, regions, half_out_size = 120, mri=False):     
    ####### grab files that were preprocessed 
    mri_files = [pos_mri for pos_mri in sorted(os.listdir(preprocess_fixed_dest)) if pos_mri.endswith('.jpg') ]
    
    hist_case   = []
    mri_case    = []
    mri_highRes = []
    mri_mask    = []
    
    if mri:
        hist_case  = getFiles(preprocess_moving_dest, 'mriUncropped', sid)
        
    else:
        hist_case  = getFiles(preprocess_moving_dest, 'hist', sid)
        
    cases = {}
    for region in regions:
        cases[region] = getFiles(preprocess_moving_dest, region, sid)
    
       
    print('Regions in registration:')
    for region in regions:
        print(region)

    for mri_file in mri_files: 
        if sid in mri_file: 
            if 'Uncropped_' in mri_file: 
                mri_highRes.append(mri_file)
            elif 'mriMask' in mri_file: 
                mri_mask.append(mri_file)
            else: 
                mri_case.append(mri_file)

    
    w, h, _ = (cv2.imread(preprocess_fixed_dest + mri_highRes[0])).shape
    count = min(len(hist_case), len(mri_case))
    
    padding_factor = int(round(max(np.add(coord[sid]['h'],np.multiply(2,coord[sid]['y_offset'])))/(coord[sid]['h'][0]+2*coord[sid]['y_offset'][0])))

    y_s = (half_out_size*(2+2*padding_factor))/h
    x_s = (half_out_size*(2+2*padding_factor))/w  
    
    out3Dhist       = np.zeros((count, half_out_size*(2+2*padding_factor), half_out_size*(2+2*padding_factor), 3))
    out3Dmri        = np.zeros((count, w, h, 3))
    out3Dmri_mask   = np.zeros((count, w, h, 3)[:-1])

    out3D_regions = {}
    for region in regions:
        out3D_regions[region] = np.zeros((count, half_out_size*(2+2*padding_factor), half_out_size*(2+2*padding_factor)))
    
    all_transforms = {}
    for idx in range(count): 
        source_image_path = preprocess_moving_dest + hist_case[idx]
        target_image_path = preprocess_fixed_dest  + mri_case[idx]
        
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

        imMri_highRes   = cv2.imread(preprocess_fixed_dest  + mri_highRes[idx])
        imMriMask       = cv2.imread(preprocess_fixed_dest  + mri_mask[idx])
        imHisto         = {}
        for region in regions:
            imHisto[region] = cv2.imread(preprocess_moving_dest + cases[region][idx])
        
        out3Dmri[idx, :, :,:]    = np.uint8(imMri_highRes)
        out3Dmri_mask[idx, :, :] = np.uint8((imMriMask[:, :, 0] > 255/2.0))

        ######## REGISTER
        affTps, regions_aff_tps, transform = runCnn(model_cache, source_image_path, target_image_path, imHisto, mri=mri) 
        all_transforms[idx] = transform
        
        # Transform main histology & regions to MRI space    
        affTps = cv2.resize(affTps*255, (h_new, w_new), interpolation=cv2.INTER_CUBIC)  
        for region in regions:
            regions_aff_tps[region] = cv2.resize(regions_aff_tps[region]*255, (h_new, w_new), interpolation=cv2.INTER_CUBIC)

            if region != 'cancer':
                regions_aff_tps[region] = regions_aff_tps[region] >255/1.5
            
        ## Create mask for warped histology
        mask_image3d  = np.zeros((affTps.shape[0], affTps.shape[1], 3), dtype=int)
        for i in range(3):
            if mri:
                mask_image3d[:, :, i]  = regions_aff_tps['Mask'][:, :, 0]
            else:
                mask_image3d[:, :, i]  = regions_aff_tps['mask'][:, :, 0]
        
        # Output histology 
        affTps = np.multiply(affTps,mask_image3d)   
        out3Dhist[idx, start_x:end_x, start_y:end_y, :] = np.uint8(affTps[:, :, :])
        
        # Output histology regions
        for region in regions:
            out3D_regions[region][idx, start_x:end_x, start_y:end_y] = np.uint8(regions_aff_tps[region][:, :, 0])

    output3D_cache = (out3Dhist, out3Dmri, out3D_regions, out3Dmri_mask, [x_s,y_s], all_transforms)
    
    return output3D_cache
    