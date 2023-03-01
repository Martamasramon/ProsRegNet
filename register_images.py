from __future__ import print_function, division
import os
import argparse
import torch
from model.ProsRegNet_model import ProsRegNet
from image.normalization import normalize_image
from geotnf.transformation import GeometricTnf
from geotnf.transformation_high_res import GeometricTnf_high_res
from geotnf.point_tnf import *
from skimage import io
import warnings
from collections import OrderedDict
import cv2
warnings.filterwarnings('ignore')
import SimpleITK as sitk
import sys
sys.path.insert(0, '../parse_data/parse_json')
from parse_registration_json import ParserRegistrationJson
from parse_study_dict import ParserStudyDict
import time 
import json
import numpy as np
from preprocess import *


tr = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

half_out_size = 500

def preprocess_image(image, high_res=False):
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


def load_models(feature_extraction_cnn, model_aff_path, model_tps_path, do_deformable=True): 
    """ 
    Load pre-trained models
    """
    
    use_cuda = torch.cuda.is_available()
    
    do_aff = not model_aff_path==''
    # do_tps = not model_tps_path==''
    do_tps = do_deformable

    # Create model
    print('Creating CNN model...')
    if do_aff:
        model_aff = ProsRegNet(use_cuda=use_cuda,geometric_model='affine',feature_extraction_cnn=feature_extraction_cnn)
    if do_tps:
        model_tps = ProsRegNet(use_cuda=use_cuda,geometric_model='tps',feature_extraction_cnn=feature_extraction_cnn)

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


def runCnn(model_cache, source_image_path, target_image_path, regions):
    """
    Run the cnn on our images and return 3D images
    """
    
    model_aff, model_tps, do_aff, do_tps, use_cuda = model_cache
    
    tpsTnf = GeometricTnf(geometric_model='tps', use_cuda=use_cuda)
    affTnf = GeometricTnf(geometric_model='affine', use_cuda=use_cuda)
    
    tpsTnf_high_res = GeometricTnf_high_res(geometric_model='tps', use_cuda=use_cuda)
    affTnf_high_res = GeometricTnf_high_res(geometric_model='affine', use_cuda=use_cuda)
    
    source_image = io.imread(source_image_path)
    target_image = io.imread(target_image_path)
    

    # copy MRI image to 3 channels 
    target_image3d = np.zeros((target_image.shape[0], target_image.shape[1], 3), dtype=int)
    target_image3d[:, :, 0] = target_image
    target_image3d[:, :, 1] = target_image
    target_image3d[:, :, 2] = target_image
    target_image = np.copy(target_image3d)


    ##### Preprocess masks 
    source_image_mask = np.copy(source_image)
    source_image_mask[np.any(source_image_mask > 5, axis=-1)] = 255
    target_image_mask = np.copy(target_image)
    target_image_mask[np.any(target_image_mask > 5, axis=-1)] = 255
    source_image_mask_var = preprocess_image(source_image_mask)
    target_image_mask_var = preprocess_image(target_image_mask)
    
    if use_cuda:
        source_image_mask_var = source_image_mask_var.cuda()
        target_image_mask_var = target_image_mask_var.cuda()
    
    batch_mask = {'source_image': source_image_mask_var, 'target_image':target_image_mask_var}


    #### Preprocess full images
    source_image_var = preprocess_image(source_image)
    target_image_var = preprocess_image(target_image)
    source_image_var_high_res = preprocess_image(source_image, high_res=True)
    target_image_var_high_res = preprocess_image(target_image, high_res=True)

    histo_image_var  = {}
    histo_image_var_high_res  = {}
    for region in regions:
        histo_image_var[region]          = preprocess_image(regions[region])
        histo_image_var_high_res[region] = preprocess_image(regions[region], high_res=True)
            
    if use_cuda:
        source_image_var = source_image_var.cuda()
        target_image_var = target_image_var.cuda()
        source_image_var_high_res = source_image_var_high_res.cuda()
        target_image_var_high_res = target_image_var_high_res.cuda()
        
        for region in regions:
            histo_image_var[region] = histo_image_var[region].cuda()
            histo_image_var_high_res[region] = histo_image_var_high_res[region].cuda()

    batch = {'source_image': source_image_var, 'target_image':target_image_var}
    batch_high_res = {'source_image': source_image_var_high_res, 'target_image':target_image_var_high_res}


    ##### Evaluate models
    if do_aff:
        model_aff.eval()
    if do_tps:
        model_tps.eval()

    if do_aff:
        ######## Affine registration using the masks only -> TWICE
        
        ## 1. 
        theta_aff=model_aff(batch_mask)
        
        warped_image_aff_high_res   = affTnf_high_res(batch_high_res['source_image'], theta_aff.view(-1,2,3))
        warped_image_aff            = affTnf(batch['source_image'], theta_aff.view(-1,2,3))
        
        warped_aff_high_res  = {}
        for region in regions:
            warped_aff_high_res[region] = affTnf_high_res(histo_image_var_high_res[region], theta_aff.view(-1,2,3))
        
        ## 2. 
        warped_mask_aff = affTnf(source_image_mask_var, theta_aff.view(-1,2,3))
        theta_aff       = model_aff({'source_image': warped_mask_aff, 'target_image': target_image_mask_var})
        
        warped_image_aff_high_res   = affTnf_high_res(warped_image_aff_high_res, theta_aff.view(-1,2,3))
        warped_image_aff            = affTnf(warped_image_aff, theta_aff.view(-1,2,3))

        for region in regions:
            warped_aff_high_res[region] = affTnf_high_res(warped_aff_high_res[region], theta_aff.view(-1,2,3))
    

    if do_aff and do_tps:
        ######## TPS registration using the images
        theta_aff_tps                   = model_tps({'source_image': warped_image_aff, 'target_image': batch['target_image']})   
        warped_image_aff_tps_high_res   = tpsTnf_high_res(warped_image_aff_high_res,theta_aff_tps)
        
        warped_aff_tps_high_res  = {}
        for region in regions:
            warped_aff_tps_high_res[region] = tpsTnf_high_res(warped_aff_high_res[region], theta_aff_tps)


    # Un-normalize images and convert to numpy
    if do_aff:
        warped_image_aff_np_high_res = normalize_image(warped_image_aff_high_res,forward=False).data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()
        warped_aff_np_high_res  = {}
        for region in regions:
            warped_aff_np_high_res[region] = normalize_image(warped_aff_high_res[region],forward=False).data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy() 

    if do_aff and do_tps:
        warped_image_aff_tps_np_high_res = normalize_image(warped_image_aff_tps_high_res,forward=False).data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()
        warped_aff_tps_np_high_res  = {}
        for region in regions:
            warped_aff_tps_np_high_res[region] = normalize_image(warped_aff_tps_high_res[region],forward=False).data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()
    
    # Ignore negative values
    warped_image_aff_np_high_res[warped_image_aff_np_high_res < 0] = 0
    warped_image_aff_tps_np_high_res[warped_image_aff_tps_np_high_res < 0] = 0    
    
    return warped_image_aff_tps_np_high_res, warped_aff_tps_np_high_res



def output_results(outputPath, inputStack, sid, fn, imSpatialInfo, extension = "nii.gz"):
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
    #sitkIm.SetDirection(tr)
    sitk.WriteImage(sitkIm, outputPath + sid + '/' + sid + fn + extension)



def getFiles(file_dest, keyword, sid): 
    cases = []
    files = [pos for pos in sorted(os.listdir(file_dest)) if keyword in pos]

    for f in files: 
        if sid in f: 
            cases.append(f)
            
    return cases
    
    
    
def register(preprocess_moving_dest, preprocess_fixed_dest, coord, model_cache, sid, hist_data): 
    for slice in hist_data:
        regions = hist_data[slice]['regions']
        break
    
    
    ####### grab files that were preprocessed 
    mri_files = [pos_mri for pos_mri in sorted(os.listdir(preprocess_fixed_dest)) if pos_mri.endswith('.jpg') ]
    
    hist_case   = []
    mri_case    = []
    mri_highRes = []
    mri_mask    = []
    
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
        
    
    volumeShape_highRes = (count, half_out_size*(2+2*padding_factor), half_out_size*(2+2*padding_factor), 3)
    out3Dhist_highRes = np.zeros(volumeShape_highRes)
    out3Dmri_highRes  = np.zeros((count, w, h, 3))
    out3Dmri_mask     = np.zeros((count, w, h, 3)[:-1])

    out3D = {}
    for region in regions:
        out3D[region] = np.zeros(volumeShape_highRes[:-1])

    ###### START ALIGNMENT
    for idx in range(count): 
        source_image_path= preprocess_moving_dest + hist_case[idx]
        target_image_path= preprocess_fixed_dest + mri_case[idx]
        
        x = coord[sid]['x'][0]
        y = coord[sid]['y'][0]
        x_offset = coord[sid]['x_offset'][0]
        y_offset = coord[sid]['y_offset'][0]
        h = coord[sid]['h'][0]
        w = coord[sid]['w'][0]
        
        
        y_s = (h+2*y_offset)/(half_out_size*2)
        x_s = (w+2*x_offset)/(half_out_size*2)
        
        
        x_prime = coord[sid]['x'][idx]
        y_prime = coord[sid]['y'][idx]
        x_offset_prime = coord[sid]['x_offset'][idx]
        y_offset_prime = coord[sid]['y_offset'][idx]
        h_prime = coord[sid]['h'][idx]
        w_prime = coord[sid]['w'][idx]
        
        w_new = (w_prime + 2*x_offset_prime)/x_s
        h_new = (h_prime + 2*y_offset_prime)/y_s
        
        start_y = int(padding_factor*half_out_size + (y_prime - y_offset_prime - y + y_offset)/y_s)
        start_x = int(padding_factor*half_out_size + (x_prime - x_offset_prime - x + x_offset)/x_s)


        imMri_highRes   = cv2.imread(preprocess_fixed_dest  + mri_highRes[idx])
        imMriMask       = cv2.imread(preprocess_fixed_dest  + mri_mask[idx])
        imHisto         = {}
        for region in regions:
            imHisto[region] = cv2.imread(preprocess_moving_dest + cases[region][idx])
        
        out3Dmri_highRes[idx, :, :,:]   = np.uint8(imMri_highRes)
        out3Dmri_mask[idx, :, :]        = np.uint8((imMriMask[:, :, 0] > 255/2.0))
       

        ######## REGISTER
        affTps, regions_aff_tps = runCnn(model_cache, source_image_path, target_image_path, imHisto) 

        for region in regions:
            regions_aff_tps[region] = cv2.resize(regions_aff_tps[region]*255, (int(h_new),  int(w_new)), interpolation=cv2.INTER_CUBIC)
            if region != 'cancer':
                regions_aff_tps[region] = regions_aff_tps[region] >255/1.5
                
            out3D[region][idx, start_x:regions_aff_tps[region].shape[0]+start_x, start_y:regions_aff_tps[region].shape[1]+start_y] = np.uint8(regions_aff_tps[region][:, :,0])
        
        affTps = cv2.resize(affTps*255, (int(h_new),  int(w_new)), interpolation=cv2.INTER_CUBIC)   

        mask_image3d = np.zeros((affTps.shape[0], affTps.shape[1], 3), dtype=int)
        for i in range(3):
            mask_image3d[:, :, i] = regions_aff_tps['mask'][:, :,0]
        
        points = np.argwhere(mask_image3d == 0)
        
        for x in range(0,points.shape[0]):
            affTps[tuple(points[x])] = 0
        out3Dhist_highRes[idx, start_x:affTps.shape[0]+start_x, start_y:affTps.shape[1]+start_y,:] = np.uint8(affTps[:, :,:])
        
        
    output3D_cache = (out3Dhist_highRes, out3Dmri_highRes, out3D, out3Dmri_mask)
    
    return output3D_cache
    
    

def main():
    """
    Entire pipeline together with preprocessing, registration, and outputting results
    """
    
    ###### INPUTS
    parser = argparse.ArgumentParser(description='Parse data')
    parser.add_argument('-v','--verbose', action='store_true',
        help='verbose output')
    
    parser.add_argument('-i','--in_path', type=str, required=True, 
        default=".",help="json file")
    
    parser.add_argument('-pm','--preprocess_moving',  action='store_true',
        help='preprocess moving')
    
    parser.add_argument('-pf','--preprocess_fixed',  action='store_true',
        help='preprocess fixed')
    
    parser.add_argument('-r','--register',  action='store_true',
        help='run deep learning registration')
    
    parser.add_argument('-e','--extension', type=str, required=False, 
        default=".", help="extension to save registered volumes (default: nii.gz)")
    
    opt = parser.parse_args()
    

    verbose = opt.verbose
    preprocess_moving = opt.preprocess_moving
    preprocess_fixed = opt.preprocess_fixed
    run_registration = opt.register
    
    timings = {}
    
    if verbose:
        print("Reading", opt.in_path)
    
    json_obj = ParserRegistrationJson(opt.in_path)
    
    if opt.extension: 
        extension = opt.extension
        print('Chosen extension is: ' + extension)
    else: 
        extension = 'nii.gz'

    try:
        with open('coord.txt') as f:
            coord = json.load(f)    
    except:
        coord = {}

    ############### START REGISTRATION HERE
    studies     = json_obj.studies
    toProcess   = json_obj.ToProcess
    outputPath  = json_obj.output_path

    ###### PREPROCESSING DESTINATIONS ######################################
    preprocess_moving_dest = outputPath + '/preprocess/hist/'
    preprocess_fixed_dest = outputPath + '/preprocess/mri/'

    # start doing preprocessing on each case and register
    for s in studies:
        if toProcess:
            if not (s in toProcess):
                print("Skipping", s)
                continue

        print("x"*30, "Processing", s,"x"*30)
        studyDict   = studies[s] 
        studyParser = ParserStudyDict(studyDict)

        sid             = studyParser.id
        fixed_img_mha   = studyParser.fixed_filename
        fixed_seg       = studyParser.fixed_segmentation_filename
        moving_dict     = studyParser.ReadMovingImage()


        ###### PREPROCESSING HISTOLOGY HERE #############################################################
        if preprocess_moving == True: 
            print('Preprocessing moving sid:', sid, '...')
            preprocess_hist(moving_dict, preprocess_moving_dest, sid)
            print('Finished preprocessing', sid)


        ###### PREPROCESSING MRI HERE #############################################################
        if preprocess_fixed == True:
            print ("Preprocessing fixed case:", sid, '...')
            coord = preprocess_mri(fixed_img_mha, fixed_seg, preprocess_fixed_dest, coord, sid)
            print("Finished processing fixed mha", sid)

            with open('coord.txt', 'w') as json_file: 
                json.dump(coord, json_file)
                
                
        ##### ALIGNMENT HERE ########################################################################
        if run_registration == True: 

            ##### LOAD MODELS
            print('.'*30, 'Begin deep learning registration for ' + sid + '.'*30)

            try:
                model_cache
            except NameError:
                feature_extraction_cnn = 'resnet101'
                
                model_aff_path = 'trained_models/best_CombinedLoss_affine_resnet101.pth.tar'
                model_tps_path = 'trained_models/best_CombinedLoss_tps_resnet101.pth.tar'
                
                model_cache = load_models(feature_extraction_cnn, model_aff_path, model_tps_path, do_deformable=True)


            ##### REGISTER
            start          = time.time()
            output3D_cache = register(preprocess_moving_dest + sid + '/' , preprocess_fixed_dest + sid + '/', coord, model_cache, sid, moving_dict)
            end            = time.time()
            
            out3Dhist_highRes, out3Dmri_highRes, out3D, out3Dmri_mask = output3D_cache
            print("Registration done in {:6.3f}(min)".format((end-start)/60.0))
            
            
            #### SAVE RESULTS
            imMri        = sitk.ReadImage(fixed_img_mha)
            imMri_array  = sitk.GetArrayFromImage(imMri)
            
            mriSpace     = imMri.GetSpacing()
            mriDirection = imMri.GetDirection()
            
            try:
                imMri_array.shape[2]
                mriOrigin = imMri[:,:,coord[sid]['slice'][0]:coord[sid]['slice'][-1]].GetOrigin()
            except:
                mriOrigin    = imMri.GetOrigin()
                mriOrigin    = (mriOrigin[0],    mriOrigin[1],    0.0)
                mriSpace     = (mriSpace[0],     mriSpace[1],     1.0)
                mriDirection = (1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0)        # This is probably not good practice, fix :)
            
            imSpatialInfo = (mriOrigin, mriSpace, mriDirection)
                
            # Write outputs as 3D volumes (.nii.gz format)
            fn_names = ['_moved.','_moved_mask.','_fixed.', '_moved_cancer.']
                
            output_results(outputPath + 'registration/', out3Dhist_highRes, sid, fn_names[0], imSpatialInfo, extension = extension)
            output_results(outputPath + 'registration/', out3D['mask'],     sid, fn_names[1], imSpatialInfo, extension = extension)
            output_results(outputPath + 'registration/', out3Dmri_highRes,  sid, fn_names[2], imSpatialInfo, extension = extension)

            # Write output for cancer segmentation if label exists
            try:
                output_results(outputPath + 'registration/', out3D['cancer'], sid, fn_names[3], imSpatialInfo, extension = extension)
            except:
                print('No cancer labels given.')

            timings[s] = (end-start)/60.0
            print('Done!')

    return timings    

if __name__=="__main__":
    timings = main()

    print("studyID",",", "Runtime (min)")
    for s in timings:
        print(s,",", timings[s])