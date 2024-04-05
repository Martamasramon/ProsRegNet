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
        "output_path": "./results/",
        
        # For the moment leave as is
        # Later change to "studies2process": studies
        
        "studies2process": {
        },
        "studies": studies
    }
    json_object = json.dumps(json_data, indent=4)
    
    # Save to file
    with open('TCIA_FUSION.json', "w") as outfile:
        outfile.write(json_object)


def main():
    folder = 'histo-invivo-VERDICT'
    # List of all available samples
    file_names = ['HMU_003_DB', 'HMU_007_TN', 'HMU_010_FH', 'HMU_011_MQ', 
                  'HMU_025_SH', 'HMU_033_JS', 'HMU_038_JC', 'HMU_056_JH',
                  'HMU_063_RS', 'HMU_065_RH', 'HMU_066_JF', 'HMU_069_NS',
                  'HMU_076_RV', 'HMU_077_MW', 'HMU_082_PS', 'HMU_084_AJ',
                  'HMU_087_FM', 'HMU_094_RB', 'HMU_099_DL', 'HMU_113_MT',
                  'HMU_116_BC', 'HMU_118_PL', 'HMU_119_MM', 'HMU_121_CN', 
                  'HMU_176_IJ', 'HMU_180_KF', 'HMU_181_MO', 'HMU_198_JL',
                  'HMU_201_MB', 'HMU_242_JD', 'HMU_245_DC', 'HMU_256_DB',
                  'HMU_258_JK', 'HMU_265_JM']

    create_json(file_names, folder)
        
main()