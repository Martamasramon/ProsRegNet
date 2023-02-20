import os
import json

def create_json(file_names):
    
    # Add all json files to 'studies'
    studies = { "aaa0069": "./jsonData/reg_aaa0069.json" }
    for name in file_names:
        studies[name] = os.path.join('./jsonData', name + '.json')
    
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
            "aaa0069": "./jsonData/reg_aaa0069.json",
            "HMU_010_FH": "./jsonData/reg_HMU_010_FH.json"
        },
        "studies": studies
    }
    json_object = json.dumps(json_data, indent=4)
    
    # Save to file
    with open('TCIA_FUSION.json', "w") as outfile:
        outfile.write(json_object)


def main():
    # List of all available samples
    file_names = ['HMU_003_DB', 'HMU_007_TN', 'HMU_010_FH', 
                'HMU_011_MQ', 'HMU_025_SH', 'HMU_038_JC', 
                'HMU_056_JH', 'HMU_064_SB', 'HMU_065_RH', 
                'HMU_066_JF', 'HMU_067_MS', 
                'HMU_069_NS', 'HMU-004-HC']

    create_json(file_names)
        
main()