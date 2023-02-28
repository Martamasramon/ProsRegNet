import os
import json

def create_json(path_mri, path_histo, file_name):
    
    json_data = {
        "id": "HMU_010_FH",
        "invivo-accession": "",
        "exvivo-accession": "",
        "fixed": os.path.join(path_mri, file_name, 'T2_nifti_' + file_name + '.nii.gz'),
        "fixed-segmentation": os.path.join(path_mri, file_name, 'T2_nifti_' + file_name + '_mask.nii.gz'),
        "fixed-landmarks2": "",
        "fixed-landmarks3": "",
        "fixed-landmarks1": "",
        "moving-type": "stack",
        "moving": os.path.join(path_histo, file_name + '.json'),
        "T2w": "",
        "ADC": "",
        "DWI": ""
    }
    json_object = json.dumps(json_data, indent=4)
    
    # Save to file
    with open('reg_' + file_name + '.json', "w") as outfile:
        outfile.write(json_object)


def main():
    # Paths to image folders 
    path_mri    = './datasets/testing/MRI'
    path_histo  = './datasets/testing/Histology'
    
    # List of all available samples
    file_names = ['HMU_003_DB', 'HMU_007_TN', 'HMU_010_FH', 
                'HMU_011_MQ', 'HMU_025_SH', 'HMU_038_JC', 
                'HMU_056_JH', 'HMU_064_SB', 'HMU_065_RH', 
                'HMU_066_JF', 'HMU_067_MS', 
                'HMU_069_NS', 'HMU-004-HC']

    for name in file_names:
        create_json(path_mri, path_histo, name)
        
main()