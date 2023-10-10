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
    
    parser.add_argument('-i',   '--in_path',   type=str, required=True,  default=".",       help="json file")
    parser.add_argument('-s',   '--save_path', type=str, required=False, default='T2-DWI',  help="folder name for registered images")
    parser.add_argument('-e',   '--extension', type=str, required=False, default="",        help="extension to save registered volumes (default: nii.gz)")
    
    # Load trained models
    parser.add_argument(      '--trained-models-dir',  type=str, default='trained_models', help='path to trained models folder')
    parser.add_argument('-n', '--trained-models-name', type=str, default='t2-dwi',         help='trained model filename')

    opt = parser.parse_args()

    verbose             = opt.verbose
    preprocess_moving   = opt.preprocess_moving
    preprocess_fixed    = opt.preprocess_fixed
    run_registration    = opt.register
    
    model_aff_path = os.path.join(opt.trained_models_dir, 'best_mri_affine.pth.tar')
    model_tps_path = os.path.join(opt.trained_models_dir, 'best_' + opt.trained_models_name + '_tps-mri.pth.tar')
    
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
        with open('coord_dwi_b90.txt') as f:
            coord_dwi = json.load(f)    
    except:
        coord_dwi = {}

    ############### START REGISTRATION HERE
    studies     = json_obj.studies
    toProcess   = json_obj.ToProcess
    outputPath  = json_obj.output_path 

    ###### PREPROCESSING DESTINATIONS ######################################
    preprocess_moving_dest = outputPath + '/preprocess/mri/'
    preprocess_fixed_dest  = outputPath + '/preprocess/dwi-b90/'

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
        fIC         = studyParser.fIC
        fIC_slice   = studyParser.fIC_slice
        cancer      = studyParser.cancer
        healthy     = studyParser.healthy

        regions = []
        if moving_seg:
            regions.append('mask')
        if cancer:
            regions.append('cancer')
        if healthy:
            regions.append('healthy')

        ###### PREPROCESSING T2 HERE #############################################################
        if preprocess_moving == True: 
            print('Preprocessing moving sid:', sid, '...')
            coord = preprocess_mri(moving_img, moving_seg, preprocess_moving_dest, coord, sid, cancer=cancer, healthy=healthy, target=False)
            print('Finished preprocessing', sid)
            
            with open('coord.txt', 'w') as json_file: 
                json.dump(coord, json_file)


        ###### PREPROCESSING DWI HERE #############################################################
        if preprocess_fixed == True:
            print ("Preprocessing fixed case:", sid, '...')
            coord_dwi = preprocess_mri(fixed_img, fixed_seg, preprocess_fixed_dest, coord_dwi, sid, fIC=fIC, dwi_map=fIC, fIC_slice=fIC_slice)
            print("Finished processing fixed mha", sid)

            with open('coord_dwi_b90.txt', 'w') as json_file: 
                json.dump(coord_dwi, json_file)
                
                
        ##### ALIGNMENT HERE ########################################################################
        if run_registration == True: 

            ##### LOAD MODELS
            print('\n' + '.'*10, 'Begin deep learning registration for ' + sid + '.'*10)
            print('Using trained model: ' + opt.trained_models_name)

            model_cache = load_models(model_aff_path, model_tps_path,do_deformable=True,tps_type='tps-mri',mri=True)

            ##### REGISTER
            start          = time.time()
            output3D_cache = register(preprocess_moving_dest + sid + '/' , preprocess_fixed_dest + sid + '/', coord_dwi, model_cache, sid, regions, half_out_size=40, mri=True, fIC=fIC)
            end            = time.time()
            
            out3D_T2, out3D_DWI, _, out3D_T2_regions, out3D_DWI_mask, scaling, transforms, _, _ = output3D_cache

            print("\nRegistration done in {:6.3f}(min)".format((end-start)/60.0))
            
            #### CALCULATE ERROR
            calc_dice(out3D_T2_regions['mask'], out3D_DWI_mask)
            hausdorff(out3D_T2_regions['mask'], out3D_DWI_mask)

            
            #### SAVE RESULTS
            imDWI           = sitk.ReadImage(fixed_img)
            if fIC:
                imDWI       = sitk.ReadImage(fIC)
            
            DWISpace        = imDWI.GetSpacing()
            T2Space         = [DWISpace[0]/scaling[0], DWISpace[1]/scaling[1], DWISpace[2]]
            mriDirection    = imDWI.GetDirection()
            mriOrigin       = imDWI[:,:,coord_dwi[sid]['slice'][0]:coord_dwi[sid]['slice'][-1]].GetOrigin()
            
            dwiSpatialInfo  = (mriOrigin, DWISpace, mriDirection)
            t2SpatialInfo   = (mriOrigin, T2Space, mriDirection)
            
            # Write outputs as 3D volumes (.nii.gz format)   
            save_path = outputPath + 'registration/' + opt.save_path + '/'       
            
            # T2W    
            output_results(save_path, out3D_T2,  sid, '_b90_moved.',      t2SpatialInfo,  model=opt.trained_models_name, extension = extension)
            for region in regions:
                output_results(save_path, out3D_T2_regions[region], sid, '_b90_moved_'+region+'.', t2SpatialInfo,  model=opt.trained_models_name, extension = extension)
            
            if fIC:
                output_results(save_path, out3D_T2_regions['fIC'], sid, '_fIC_moved.' , t2SpatialInfo, model=opt.trained_models_name, extension = extension)
                for region in regions:
                    output_results(save_path, out3D_T2_regions['fIC_'+region], sid, '_fIC_moved_' + region + '.' , t2SpatialInfo, model=opt.trained_models_name, extension = extension)
                
            # DWI 
            output_results(save_path, out3D_DWI,                sid, '_b90_fixed.',      dwiSpatialInfo, model=opt.trained_models_name, extension = extension)
            output_results(save_path, out3D_DWI_mask,           sid, '_b90_fixed_mask.', dwiSpatialInfo, model=opt.trained_models_name, extension = extension)
            
            if fIC:
                out_map  = get_map(preprocess_fixed_dest + sid + '/', 'fIC')
                output_results(save_path, out_map,        sid, '_fIC_fixed.',      dwiSpatialInfo,   model=opt.trained_models_name, extension = extension)
                output_results(save_path, out3D_DWI_mask, sid, '_fIC_fixed_mask.', dwiSpatialInfo,   model=opt.trained_models_name, extension = extension)
                
            
            save_all_transforms(transforms, sid, dwiSpatialInfo, scaling, '/T2-DWI/')

            timings[s] = (end-start)/60.0
            print('Done!')

    return timings    

if __name__=="__main__":
    timings = main()

    print("\nstudyID",",", "Runtime (min)")
    for s in timings:
        print(s,",", timings[s])