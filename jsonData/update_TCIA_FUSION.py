import os
import json

def create_json(file_names,folder):
    
    # Add all json files to 'studies'
    studies = {}
    for name in file_names:
        studies[name] = os.path.join('./jsonData', folder, 'reg_' + name + '.json')
    
    json_data = {
        "version": "0.1",
        "type": "registration",
        "method": {
            "type": "3DRegistration",
            "params": "-da -dd"
        },
        "output_path": "/cluster/project7/backup_masramon/Registration results/",
        "studies2process": studies,
        "studies":         studies
    }
    json_object = json.dumps(json_data, indent=4)
    
    # Save to file
    with open('TCIA_FUSION_t2w.json', "w") as outfile:
        outfile.write(json_object)


def main():
    folder = 'histo-invivo-mpMRI'
    # List of all available samples
    file_names = ['HMU_010_FH', 'HMU_038_JC', 'HMU_063_RS', 'HMU_066_JF', 'HMU_076_RV', 'HMU_082_PS', 
                'HMU_084_AJ','HMU_113_MT', 'HMU_121_CN', 'HMU_176_IJ', 'HMU_180_KF', 'HMU_201_MB',
                'HMU_003_DB', 'HMU_007_TN','HMU_011_MQ','HMU_033_JS','HMU_056_JH','HMU_060_CH',
                'HMU_065_RH','HMU_067_MS','HMU_068_PB','HMU_077_MW','HMU_087_FM','HMU_094_RB', 'HMU_099_DL', 
                'HMU_112_AD','HMU_116_BC','HMU_118_PL', 'HMU_119_MM','HMU_128_RK','HMU_174_IS', 'HMU_181_MO',
                'HMU_198_JL', 'HMU_226_NS','HMU_227_KT','HMU_235_CC', 'HMU_242_JD', 'HMU_245_DC', 'HMU_256_DB',
                'HMU_265_JM', 'HMU_342_ME','HMU_348_EA']    
    
    create_json(file_names, folder)
        
main()