import os
import json

def create_json(path_mri, name):
    
    json_data = {
        "id":                   name + '_T2_DWI',
        "moving":               os.path.join(path_mri, 'MRI', name, 'MRI_in_vivo_' + name + '.nii.gz'),
        "moving-segmentation":  os.path.join(path_mri, 'MRI', name, 'MRI_in_vivo_' + name + '_mask.nii.gz'),
        "fixed":                os.path.join(path_mri, 'DWI', name, 'DWI_' + name + '.nii.gz'),
        "fixed-segmentation":   os.path.join(path_mri, 'DWI', name, 'DWI_' + name + '_mask.nii.gz')
    }
    json_object = json.dumps(json_data, indent=4)
    
    # Save to file
    with open('reg_T2_DWI_' + name + '.json', "w") as outfile:
        outfile.write(json_object)


def main():
    # Paths to image folders 
    path_mri    = './datasets/testing/'
                
    file_names = ['HMU_003_DB', 'HMU_007_TN', 'HMU_010_FH']

    for name in file_names:
        create_json(path_mri, name)
        
main()