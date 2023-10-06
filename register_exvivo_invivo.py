from __future__ import print_function, division
import os
import json
import argparse
import sys
import time 
import json
import SimpleITK            as      sitk
from register_functions     import  *

sys.path.insert(0, '../parse_data/parse_json')
from parse_registration_json    import ParserRegistrationJson
from parse_study_dict           import ParserStudyDict

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
    
    parser.add_argument('-i',   '--in_path',   type=str, required=True,  default=".",               help="json file")
    parser.add_argument('-s',   '--save_path', type=str, required=False, default='exvivo-invivo',   help="folder name for registered images")
    parser.add_argument('-e',   '--extension', type=str, required=False, default="",                help="extension to save registered volumes (default: nii.gz)")
    
    # Load trained models
    parser.add_argument(      '--trained-models-dir',  type=str, default='trained_models', help='path to trained models folder')
    parser.add_argument('-n', '--trained-models-name', type=str, default='t2-dwi',         help='trained model filename')

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

    try:
        with open('coord.txt') as f:
            coord = json.load(f)    
    except:
        coord = {}
    try:
        with open('coord_exvivo.txt') as f:
            coord_exvivo = json.load(f)    
    except:
        coord_exvivo = {}

    ############### START REGISTRATION HERE
    studies     = json_obj.studies
    toProcess   = json_obj.ToProcess
    outputPath  = json_obj.output_path 

    ###### PREPROCESSING DESTINATIONS ######################################
    preprocess_moving_dest = outputPath + '/preprocess/exvivo/'
    preprocess_fixed_dest  = outputPath + '/preprocess/mri/'

    # start doing preprocessing on each case and register
    for s in studies:
        if toProcess:
            if not (s in toProcess):
                print("Skipping", s)
                continue

        print('\n',"x"*30, "Processing", s,"x"*30)
        studyDict   = studies[s] 
        studyParser = ParserStudyDict(studyDict)

        sid         = studyParser.id
        fixed_img   = studyParser.fixed_filename
        fixed_seg   = studyParser.fixed_segmentation_filename
        moving_img  = studyParser.moving_filename
        moving_seg  = studyParser.moving_segmentation_filename

        regions = ['Mask']

        ###### PREPROCESSING EX-VIVO IMAGES HERE #############################################################
        if preprocess_moving == True: 
            print('Preprocessing moving sid:', sid, '...')
            coord_exvivo = preprocess_mri(moving_img, moving_seg, preprocess_moving_dest, coord_exvivo, sid, crop_mask=True, exvivo=True)
            print('Finished preprocessing moving image (ex-vivo)', sid)
            
            with open('coord_exvivo.txt', 'w') as json_file: 
                json.dump(coord_exvivo, json_file)


        ###### PREPROCESSING IN-VIVO IMAGES HERE #############################################################
        if preprocess_fixed == True:
            print ("Preprocessing fixed case:", sid, '...')
            coord = preprocess_mri(fixed_img, fixed_seg, preprocess_fixed_dest, coord, sid)
            print("Finished processing fixed image (in-vivo)", sid)

            with open('coord.txt', 'w') as json_file: 
                json.dump(coord, json_file)
                
                
        ##### ALIGNMENT HERE ########################################################################
        if run_registration == True: 

            ##### LOAD MODELS
            print('.'*10, 'Begin deep learning registration for ' + sid + '.'*10)
            print('Using trained model: ' + opt.trained_models_name)

            try:
                model_cache
            except:
                model_cache = load_models(model_aff_path, model_tps_path, do_deformable=True)

            ##### REGISTER
            start          = time.time()
            output3D_cache = register(preprocess_moving_dest + sid + '/' , preprocess_fixed_dest + sid + '/', coord, model_cache, sid, regions, mri=True, exvivo=True)
            end            = time.time()
            
            out3D_exvivo, out3D_invivo, _, out3D_exvivo_regions, out3D_invivo_mask, scaling, transforms, _, _ = output3D_cache
            print("Registration done in {:6.3f}(min)".format((end-start)/60.0))
            
            #### CALCULATE ERROR
            calc_dice(out3D_exvivo_regions['Mask'], out3D_invivo_mask)
            hausdorff(out3D_exvivo_regions['Mask'], out3D_invivo_mask)

            #### SAVE RESULTS
            image_invivo    = sitk.ReadImage(fixed_img)
            
            invivo_space        = image_invivo.GetSpacing()
            exvivo_space        = [invivo_space[0]/scaling[0], invivo_space[1]/scaling[1], invivo_space[2]]
            mriDirection        = image_invivo.GetDirection()
            mriOrigin           = image_invivo[:,:,coord[sid]['slice'][0]:coord[sid]['slice'][-1]].GetOrigin()
            invivo_SpatialInfo  = (mriOrigin, invivo_space, mriDirection)
            exvivo_SpatialInfo  = (mriOrigin, exvivo_space, mriDirection)
            
            # Write outputs as 3D volumes (.nii.gz format)   
            save_path = outputPath + 'registration/' + opt.save_path + '/'                    
            output_results(save_path, out3D_invivo,                 sid, '_fixed.',      invivo_SpatialInfo,  model=opt.trained_models_name, extension = extension)
            output_results(save_path, out3D_invivo_mask,            sid, '_fixed_mask.', invivo_SpatialInfo,  model=opt.trained_models_name, extension = extension)
            output_results(save_path, out3D_exvivo,                 sid, '_moved.',      exvivo_SpatialInfo,  model=opt.trained_models_name, extension = extension)
            output_results(save_path, out3D_exvivo_regions['Mask'], sid, '_moved_mask.', exvivo_SpatialInfo,  model=opt.trained_models_name, extension = extension)
            
            save_all_transforms(transforms, sid, invivo_SpatialInfo, scaling, '/exvivo_invivo/')

            timings[s] = (end-start)/60.0
            print('Done!')

    return timings    

if __name__=="__main__":
    timings = main()

    print("studyID",",", "Runtime (min)")
    for s in timings:
        print(s,",", timings[s])