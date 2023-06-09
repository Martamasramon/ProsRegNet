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
        moving_dict     = studyParser.ReadMovingImage()
        dwi             = studyParser.DWI
        fIC             = studyParser.fIC

        for slice in moving_dict:
            regions = moving_dict[slice]['regions']
            break
        
        ###### PREPROCESSING DESTINATIONS ######################################
        preprocess_moving_dest = outputPath + '/preprocess/hist/'
        if dwi:
            preprocess_fixed_dest  = outputPath + '/preprocess/dwi/'
            coord_path             = 'coord_dwi.txt'
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
            preprocess_hist(moving_dict, preprocess_moving_dest, sid, fIC=fIC)
            print('Finished preprocessing', sid)


        ###### PREPROCESSING MRI HERE #############################################################
        if preprocess_fixed == True:
            print ("Preprocessing fixed case:", sid, '...')
            coord = preprocess_mri(fixed_img_mha, fixed_seg, preprocess_fixed_dest, coord, sid, fIC=fIC)
            print("Finished processing fixed mha", sid)

            with open(coord_path, 'w') as json_file: 
                json.dump(coord, json_file)
                
                
        ##### ALIGNMENT HERE ########################################################################
        if run_registration == True: 

            ##### LOAD MODELS
            print('.'*10, 'Begin deep learning registration for ' + sid + '.'*10)
            print('Using trained model: ' + opt.trained_models_name)

            try:
                model_cache
            except NameError:
                model_cache = load_models(model_aff_path, model_tps_path, do_deformable=True)

            ##### REGISTER
            start          = time.time()
            output3D_cache = register(preprocess_moving_dest + sid + '/' , preprocess_fixed_dest + sid + '/', coord, model_cache, sid, regions, fIC=fIC)
            end            = time.time()
            
            out3Dhist, out3Dmri, out3Dhist_regions, out3Dmri_mask, scaling, transforms = output3D_cache
            print("Registration done in {:6.3f}(min)".format((end-start)/60.0))
            
            #### CALCULATE DICE
            calc_dice(out3Dhist_regions['mask'], out3Dmri_mask)
            
            #### SAVE RESULTS
            imMri           = sitk.ReadImage(fixed_img_mha)
            if fIC:
                imMri       = sitk.ReadImage(fIC)
                
            mriSpace        = imMri.GetSpacing()
            histSpace       = [mriSpace[0]/scaling[0], mriSpace[1]/scaling[1], mriSpace[2]]
            mriDirection    = imMri.GetDirection()
            mriOrigin       = imMri[:,:,coord[sid]['slice'][0]:coord[sid]['slice'][-1]].GetOrigin()
            imSpatialInfo   = (mriOrigin, mriSpace, mriDirection)
            histSpatialInfo = (mriOrigin, histSpace, mriDirection)
            
            #direction       = ( x ,0.0,0.0,0.0, y ,0.0,0.0,0.0, z )

            # Write outputs as 3D volumes (.nii.gz format)   
            if dwi:
                folder = 'histo-DWI/'                
            else:
                folder = 'histo-T2/' 
            save_path = outputPath + 'registration/' + folder
            
            output_results(save_path, out3Dmri,  sid, '_fixed.', imSpatialInfo,   model=opt.trained_models_name, extension = extension)
            output_results(save_path, out3Dhist, sid, '_moved.', histSpatialInfo, model=opt.trained_models_name, extension = extension)
            for region in regions:
                output_results(save_path, out3Dhist_regions[region], sid, '_moved_' + region + '.' , histSpatialInfo, model=opt.trained_models_name, extension = extension)
            
            if fIC:
                output_results(save_path, out3Dhist_regions['fIC'], sid, '_moved_fIC.' , histSpatialInfo, model=opt.trained_models_name, extension = extension)
                out_fIC     = get_fIC(preprocess_fixed_dest + sid + '/')
                output_results(save_path, out_fIC,  sid, '_fIC.', imSpatialInfo,   model=opt.trained_models_name, extension = extension)
                
            save_all_transforms(transforms, sid, imSpatialInfo, scaling, folder)

            timings[s] = (end-start)/60.0
            print('Done!')

    return timings    

if __name__=="__main__":
    timings = main()

    print("studyID",",", "Runtime (min)")
    for s in timings:
        print(s,",", timings[s])