from __future__ import print_function, division
import os
import json
import argparse
import sys
import time 
import json
import SimpleITK    as sitk

from geotnf.point_tnf           import *
from process_img                import *
from preprocess                 import *
from register_functions         import *
from landmark_functions         import *

sys.path.insert(0, '../parse_data/parse_json')
from parse_registration_json import ParserRegistrationJson
from parse_study_dict import ParserStudyDict

import warnings
warnings.filterwarnings('ignore')


def main():
    """
    Entire pipeline together with preprocessing, registration, and outputting results
    """
    
    ###### INPUTS
    parser = argparse.ArgumentParser(description='Parse data')
    parser.add_argument('-v',   '--verbose',            action='store_true', help='verbose output')
    parser.add_argument('-pm',  '--preprocess_moving',  action='store_true', help='preprocess moving')
    parser.add_argument('-pf',  '--preprocess_fixed',   action='store_true', help='preprocess fixed')
    parser.add_argument('-r',   '--register',           action='store_true', help='run deep learning registration')
    
    parser.add_argument('-i',   '--in_path',   type=str, required=True,  default=".",           help="json file")
    parser.add_argument('-e',   '--extension', type=str, required=False, default="",            help="extension to save registered volumes (default: nii.gz)")
    
    # Load trained models
    parser.add_argument(      '--trained-models-dir',  type=str, default='trained_models', help='path to trained models folder')
    parser.add_argument('-n', '--trained-models-name', type=str, default='default',        help='trained model filename')

    opt = parser.parse_args()
    

    verbose             = opt.verbose
    preprocess_moving   = opt.preprocess_moving
    preprocess_fixed    = opt.preprocess_fixed
    run_registration    = opt.register
    
    model_aff_path = os.path.join(opt.trained_models_dir, 'best_default_affine.pth.tar')
    model_tps_path = os.path.join(opt.trained_models_dir, 'best_' + opt.trained_models_name + '_tps.pth.tar')
    
    timings = {}
    
    if verbose:
        print("Reading", opt.in_path)
    
    json_obj = ParserRegistrationJson(opt.in_path)
    
    if opt.extension: 
        extension = opt.extension
        print('Chosen extension is: ' + extension)
    else: 
        extension = 'nii.gz'

    ############### START REGISTRATION HERE
    studies     = json_obj.studies
    toProcess   = json_obj.ToProcess
    outputPath  = json_obj.output_path 

    # start doing preprocessing on each case and register
    for s in studies:
        if toProcess:
            if not (s in toProcess):
                print("Skipping", s)
                continue

        print('\n',"x"*30, "Processing", s,"x"*30)
        studyDict   = studies[s] 
        studyParser = ParserStudyDict(studyDict)

        sid             = studyParser.id
        fixed_img_mha   = studyParser.fixed_filename
        fixed_seg       = studyParser.fixed_segmentation_filename
        mri_cancer      = studyParser.cancer
        moving_dict     = studyParser.ReadMovingImage()
        dwi             = studyParser.DWI
        fIC             = studyParser.fIC   
        dwi_map         = studyParser.DWI_map
        landmarks       = studyParser.landmarks
        exvivo          = studyParser.exvivo
        print('exvivo', exvivo)
        
        for slice in moving_dict:
            regions = moving_dict[slice]['regions']
            break
        
        ###### PREPROCESSING DESTINATIONS ######################################
        preprocess_moving_dest = outputPath + '/preprocess/hist/'
        if dwi:
            if fIC:
                preprocess_fixed_dest  = outputPath + '/preprocess/dwi-b90/'
                coord_path             = 'coord_dwi_b90.txt'
            else:
                preprocess_fixed_dest  = outputPath + '/preprocess/dwi-b0/'
                coord_path             = 'coord_dwi_b0.txt'
        else:
            if exvivo:
                preprocess_fixed_dest  = outputPath + '/preprocess/exvivo/'
                coord_path             = 'coord_exvivo.txt'
            else:  
                preprocess_fixed_dest  = outputPath + '/preprocess/mri/'
                coord_path             = 'coord.txt'
        
        try:
            with open(coord_path) as f:
                coord = json.load(f)    
        except:
            coord = {}

        ###### PREPROCESSING HISTOLOGY HERE #############################################################
        if preprocess_moving == True: 
            print('Preprocessing moving sid:', sid, '...')
            landmarks_histo = preprocess_hist(moving_dict, preprocess_moving_dest, sid, dwi=dwi, fIC=fIC)
            print('Finished preprocessing moving image', sid)
        else:
            landmarks_histo = {}


        ###### PREPROCESSING MRI HERE #############################################################
        if preprocess_fixed == True:
            print ("Preprocessing fixed case:", sid, '...')
            coord = preprocess_mri(fixed_img_mha, fixed_seg, preprocess_fixed_dest, coord, sid, dwi_map=dwi_map, fIC=fIC, cancer=mri_cancer, landmarks=landmarks, exvivo=exvivo)
            print("Finished preprocessing fixed image", sid)

            with open(coord_path, 'w') as json_file: 
                json.dump(coord, json_file)                
                
        ##### ALIGNMENT HERE ########################################################################
        if run_registration == True: 

            ##### LOAD MODELS
            print('\n', '.'*10, 'Begin deep learning registration for ' + sid + '.'*10)
            print('Using trained model: ' + opt.trained_models_name)

            try:
                model_cache
            except:
                model_cache = load_models(model_aff_path, model_tps_path, do_deformable=True)

            ##### REGISTER
            start          = time.time()
            output3D_cache = register(preprocess_moving_dest + sid + '/' , preprocess_fixed_dest + sid + '/', coord, model_cache, sid, regions, landmarks_histo, landmarks, fIC=fIC)
            end            = time.time()
            
            out3Dhist, out3Dmri, out3Dmri_cancer, out3Dhist_regions, out3Dmri_mask, scaling, transforms, landmark_image_mri, landmark_image_histo = output3D_cache
            print("\nRegistration done in {:6.3f}(min)".format((end-start)/60.0))
            
            #### CALCULATE ERROR
            calc_dice(out3Dhist_regions['mask'], out3Dmri_mask)
            hausdorff(out3Dhist_regions['mask'], out3Dmri_mask)

            try:
                print('\nCancer...')
                if fIC:
                    fIC_cancer = out3Dmri_cancer.copy()
                    for i in range(3):
                        out3Dmri_cancer[:,:,:,i] = cv2.flip(cv2.flip(out3Dmri_cancer[:,:,:,i],0),1)

                calc_dice(out3Dhist_regions['cancer'], out3Dmri_cancer[:,:,:,0])
                hausdorff(out3Dhist_regions['cancer'], out3Dmri_cancer[:,:,:,0])
            except:
                pass
            
            #### SAVE RESULTS
            imMri           = sitk.ReadImage(fixed_img_mha)
            if fIC:
                imMri       = sitk.ReadImage(dwi_map)
                
            mriSpace        = imMri.GetSpacing()
            histSpace       = [mriSpace[0]/scaling[0], mriSpace[1]/scaling[1], mriSpace[2]]
            mriDirection    = imMri.GetDirection() 
            mriOrigin       = imMri[:,:,coord[sid]['slice'][0]:coord[sid]['slice'][-1]].GetOrigin()
            imSpatialInfo   = (mriOrigin, mriSpace, mriDirection)
            histSpatialInfo = (mriOrigin, histSpace, mriDirection)
            
            # Write outputs as 3D volumes (.nii.gz format)   
            if dwi:
                if fIC:
                    tag = '_b90'     
                else:
                    tag = '_b0'  
            else:
                if exvivo:
                    tag = '_exvivo'
                else:
                    tag = '_T2'
            save_path = outputPath + 'registration/histo-' + tag[1:] + '/'
            
            ## Output histology
            output_results(save_path, out3Dhist, sid, tag + '_moved.', histSpatialInfo, model=opt.trained_models_name, extension = extension)
            for region in regions:
                output_results(save_path, out3Dhist_regions[region], sid, tag + '_moved_' + region + '.' , histSpatialInfo, model=opt.trained_models_name, extension = extension)
            #output_results(save_path, 1-out3Dhist_regions['mask'], sid, tag + '_moved_reverse_mask.' , histSpatialInfo, model=opt.trained_models_name, extension = extension)
            
            if dwi and fIC:
                output_results(save_path, out3Dhist_regions['fIC'], sid, '_fIC_moved.' , histSpatialInfo, model=opt.trained_models_name, extension = extension)
                output_results(save_path, out3Dhist_regions['fIC-mask'], sid, '_fIC_mask_moved.' , histSpatialInfo, model=opt.trained_models_name, extension = extension)
                try:
                    output_results(save_path, out3Dhist_regions['fIC-density'], sid, '_fIC_density_moved.' , histSpatialInfo, model=opt.trained_models_name, extension = extension)
                except:
                    pass
                
            if landmarks:
                output_results(save_path, landmark_image_histo, sid, tag +'_landmarks_moved.', histSpatialInfo,   model=opt.trained_models_name, extension = extension)  
                       
            ## Output MRI 
            output_results(save_path, out3Dmri,         sid, tag +'_fixed.', imSpatialInfo,   model=opt.trained_models_name, extension = extension)
            output_results(save_path, out3Dmri_mask,    sid, tag +'_fixed_mask.', imSpatialInfo,   model=opt.trained_models_name, extension = extension)
            if dwi:
                if fIC:
                    text = 'fIC'
                else:
                    text = 'ADC'
                try:
                    out_map  = get_map(preprocess_fixed_dest + sid + '/', text)
                    output_results(save_path, out_map,  sid, '_'+text+'.', imSpatialInfo,   model=opt.trained_models_name, extension = extension)
                except:
                    print("Couldn't get DWI map")
                    
            if mri_cancer:
                output_results(save_path, out3Dmri_cancer,  sid, tag +'_cancer.', imSpatialInfo,   model=opt.trained_models_name, extension = extension)   

                if fIC:
                    output_results(save_path, fIC_cancer,  sid, '_fIC_cancer.', imSpatialInfo,   model=opt.trained_models_name, extension = extension)   

            if landmarks:
                output_results(save_path, landmark_image_mri, sid, tag +'_landmarks_fixed.', imSpatialInfo,   model=opt.trained_models_name, extension = extension)   
                 
            ## Save transforms 
            save_all_transforms(transforms, sid, imSpatialInfo, scaling, 'histo-' + tag[1:] + '/')

            timings[s] = (end-start)/60.0
            print('Done!\n')

    return timings    

if __name__=="__main__":
    timings = main()

    print("studyID",",", "Runtime (min)")
    for s in timings:
        print(s,",", timings[s])