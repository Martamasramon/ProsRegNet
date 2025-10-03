import os
import json

def create_folders(path_mri, path_histo, sid):
    try: 
        directory = os.path.join('..',path_mri,sid)
        os.mkdir(directory)
        print('Made directory', directory)
    except: 
        pass 

    try: 
        directory = os.path.join('..',path_histo,sid)
        os.mkdir(directory)
        print('Made directory', directory)
    except: 
        pass 
    
    
def create_json(path_mri, path_histo, path_json, file_name):
    
    json_data = {
        "id":                   file_name,
        "moving-type":          "stack",
        "fixed":                os.path.join(path_mri, file_name, file_name + '_T2W.nii.gz'),
        "fixed-segmentation":   os.path.join(path_mri, file_name, file_name + '_T2W_mask.nii.gz'),
        "cancer":               os.path.join(path_mri, file_name, file_name + '_T2W_urethra.nii.gz'),
        "DWI":                  "",
        "fIC":                  "",
        "DWI-map":              '',
        "moving":               os.path.join(path_histo, file_name + '_best.json')
    }
    json_object = json.dumps(json_data, indent=4)
    
    # Save to file
    with open(os.path.join(path_json, 'reg_' + file_name + '.json'), "w") as outfile:
        outfile.write(json_object)


def main():
    # Paths to image folders 
    path_mri    = '/cluster/project7/backup_masramon/MRI/in-vivo/'
    path_histo  = './datasets/testing/'
    path_json   = 'histo-invivo-mpMRI'
    
    # List of samples
    ## 'best'
    file_names = [#'HMU_003_DB','HMU_004_HC','HMU_007_TN','HMU_011_MQ','HMU_067_MS','HMU_094_RB','HMU_116_BC','HMU_128_RK','HMU_181_MO','HMU_226_NS','HMU_245_DC',
              #'HMU_258_JK','HMU_342_ME','HMU_348_EA','HMU_056_JH','HMU_118_PL', 'HMU_119_MM', 'HMU_198_JL',
              'HMU_010_FH', 'HMU_038_JC', 'HMU_063_RS', 'HMU_066_JF', 'HMU_076_RV', 'HMU_082_PS', 
                'HMU_084_AJ','HMU_113_MT', 'HMU_121_CN', 'HMU_176_IJ', 'HMU_180_KF', 'HMU_201_MB']
    
    
    # file_names = ['HMU_003_DB','HMU_004_HC','HMU_007_TN','HMU_011_MQ','HMU_067_MS','HMU_068_PB','HMU_094_RB','HMU_116_BC','HMU_128_RK','HMU_181_MO','HMU_226_NS','HMU_245_DC',
    #               'HMU_258_JK','HMU_342_ME','HMU_348_EA','HMU_010_FH', 
    # 'HMU_025_SH', 'HMU_033_JS', 'HMU_038_JC', 'HMU_056_JH',
    # 'HMU_060_CH', 'HMU_063_RS', 'HMU_065_RH', 'HMU_066_JF', 
    # 'HMU_069_NS', 'HMU_076_RV', 'HMU_077_MW', 'HMU_082_PS', 
    # 'HMU_084_AJ', 'HMU_087_FM', 'HMU_094_RB', 'HMU_099_DL','HMU_112_AD',
    # 'HMU_113_MT', 'HMU_116_BC', 'HMU_118_PL', 'HMU_119_MM', 
    # 'HMU_121_CN', 'HMU_174_IS', 'HMU_176_IJ', 'HMU_180_KF', 
    # 'HMU_181_MO', 'HMU_198_JL', 'HMU_201_MB', 'HMU_227_KT', 
    # 'HMU_235_CC', 'HMU_242_JD', 'HMU_245_DC', 'HMU_256_DB',
    # 'HMU_258_JK', 'HMU_265_JM']

    for name in file_names:
        #create_folders(path_mri, path_histo, name)
        create_json(path_mri, path_histo, path_json, name)
        
main()