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
        "fixed":                os.path.join(path_mri, file_name, file_name + '_b3000.nii.gz'),
        "fixed-segmentation":   os.path.join(path_mri, file_name, file_name + '_b3000_mask.nii.gz'),
        "cancer":               "",
        "DWI":                  "True",
        "fIC":                  "True",
        "DWI-map":              os.path.join(path_mri, file_name, file_name + '_fIC_DL.nii.gz'),
        "moving":               os.path.join(path_histo, file_name + '.json')
    }
    json_object = json.dumps(json_data, indent=4)
    
    # Save to file
    with open(os.path.join(path_json, 'reg_' + file_name + '.json'), "w") as outfile:
        outfile.write(json_object)


def main():
    # Paths to image folders 
    path_mri    = './datasets/testing/DWI'
    path_histo  = './datasets/testing/Histology'
    path_json   = 'histo-invivo-VERDICT'
    
    # List of samples
    file_names = ['HMU_033_JS','HMU_038_JC','HMU_056_JH','HMU_063_RS','HMU_065_RH','HMU_066_JF','HMU_069_NS','HMU_076_RV','HMU_077_MW','HMU_084_AJ','HMU_087_FM','HMU_094_RB','HMU_099_DL','HMU_121_CN']
    
    for name in file_names:
        create_folders(path_mri, path_histo, name)
        create_json(path_mri, path_histo, path_json, name)
        
main()