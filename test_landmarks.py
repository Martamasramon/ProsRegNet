  
"""
If you use this code, please cite the following papers:
(1) Shao, Wei, et al. "ProsRegNet: A Deep Learning Framework for Registration of MRI and Histopathology Images of the Prostate." Medical Image Analysis. 2020.
(2) Rocco, Ignacio, Relja Arandjelovic, and Josef Sivic. "Convolutional neural network architecture for geometric matching." Proceedings of CVPR. 2017.

The following code is adapted from: https://github.com/ignacio-rocco/cnngeometric_pytorch.
"""

from __future__ import print_function, division
import argparse
import os
import cv2
import torch
import numpy    as np
from torch.utils.data                   import DataLoader
from image.normalization                import NormalizeImageDict
from data.landmark_dataset              import LandmarkDataset, list_to_image, image_to_list
from geotnf.transformation_landmarks    import LandmarkTnf
from geotnf.transformation              import GeometricTnf
from geotnf.point_tnf                   import PointTnf
from register_functions                 import load_models, calc_dice

# Ignore warnings
import warnings
warnings.simplefilter("ignore", UserWarning)

def save_csv(name, data):
    np.savetxt("results/landmarks/landmarks_" + name + ".csv", torch.squeeze(data).cpu().detach().numpy(), delimiter=",")

def save_results(name, img):
    cv2.imwrite("results/landmarks/" + name + ".png", img.data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy() * 255)
    
# Argument parsing
parser = argparse.ArgumentParser(description='ProsRegNet PyTorch implementation')

# Paths
parser.add_argument('-t', '--test-csv-name',       type=str, default='landmarks_train.csv',  help='test data csv file name')

parser.add_argument('-p', '--training-image-path', type=str, default='datasets/training_landmarks/',  help='path to folder containing training images')
parser.add_argument(      '--trained-models-dir',  type=str, default='trained_models',                help='path to trained models folder')
parser.add_argument('-n', '--trained-models-name', type=str, default='default',                       help='trained model filename')

# Optimization parameters 
parser.add_argument('--batch-size',     type=int,   default=1,      help='training batch size')
parser.add_argument('--seed',           type=int,   default=1,      help='Pseudo-RNG seed')

args     = parser.parse_args()

## CUDA
use_cuda = torch.cuda.is_available() 
print("Use Cuda? ", use_cuda)

# Seed
if use_cuda:
    torch.cuda.manual_seed(args.seed)

## CNN model and loss
print('Creating CNN model...')
geometric_model = 'tps'

model_aff_path = os.path.join(args.trained_models_dir, 'best_default_affine.pth.tar')
model_tps_path = os.path.join(args.trained_models_dir, 'best_' + args.trained_models_name + '_tps.pth.tar')
    
model_aff, model_tps, _, _, _ = load_models(model_aff_path, model_tps_path)

landmark_tnf    = LandmarkTnf(use_cuda=use_cuda)
point_tnf       = PointTnf(use_cuda=use_cuda)
aff_tnf         = GeometricTnf(use_cuda=use_cuda, geometric_model='affine')
tps_tnf         = GeometricTnf(use_cuda=use_cuda, geometric_model='tps')

# Dataset and dataloader
dataset_test = LandmarkDataset(geometric_model      = geometric_model,
                            csv_file             = './training_data/tps/' + args.test_csv_name,
                            training_image_path  = args.training_image_path,
                            batch_size           = args.batch_size,
                            transform            = NormalizeImageDict(['source_image','target_image']))

dataloader_test  = DataLoader(dataset_test,  batch_size=args.batch_size, shuffle=False, num_workers=4)

model_aff.eval()
model_tps.eval()
total_landmark_loss = 0
total_dice_loss     = 0

for batch_idx, batch in enumerate(dataloader_test):
    tnf_batch       = landmark_tnf(batch)
    img_landmarks   = list_to_image((240,240), tnf_batch['source_landmarks'][0,0,:],tnf_batch['source_landmarks'][0,1,:],3,tensor=True,use_cuda=use_cuda)
    name            = tnf_batch['name'][0][:10]
    
    # Affine transformation -- 1
    input_batch = {'source_image': tnf_batch['source_mask'], 'target_image': tnf_batch['target_mask']}
    theta_aff_1 = model_aff(input_batch)
    
    warped_histo            = aff_tnf(tnf_batch['source_image'], theta_aff_1)
    warped_mask             = aff_tnf(tnf_batch['source_mask'],  theta_aff_1)
    warped_img_landmarks    = aff_tnf(img_landmarks,  theta_aff_1)
    #warped_landmarks       = point_tnf.affPointTnf(theta_aff_1, tnf_batch['source_landmarks'])
    
    # Affine transformation -- 2
    input_batch = {'source_image': warped_mask, 'target_image': tnf_batch['target_mask']}
    theta_aff_2 = model_aff(input_batch)
    
    warped_histo            = aff_tnf(warped_histo, theta_aff_2)
    warped_img_landmarks    = aff_tnf(warped_img_landmarks,  theta_aff_2)
    #warped_landmarks       = point_tnf.affPointTnf(theta_aff_2, warped_landmarks)
    
    # TPS transformation
    input_batch = {'source_image': warped_histo, 'target_image': tnf_batch['target_image']}
    theta_tps   = model_tps(input_batch)
    
    warped_histo            = tps_tnf(warped_histo, theta_tps)
    warped_img_landmarks    = tps_tnf(warped_img_landmarks,  theta_tps)
    #warped_landmarks       = point_tnf.tpsPointTnf(theta_tps, warped_landmarks)    
   
    # Transform image of landmarks to list
    warped_landmarks        = image_to_list(warped_img_landmarks, use_cuda=use_cuda)

    # Save landmark locations 
    save_landmarks = {
        "histo_":           warped_landmarks,
        #"histo_source_":    tnf_batch['source_landmarks'], 
        "mri_":             tnf_batch['target_landmarks'], 
    }
    for item in save_landmarks:
        save_csv(item + name, save_landmarks[item])

    # Save images
    save_images = {
        "histo_":           warped_histo,
        #"histo_source_":    tnf_batch['source_image'], 
        "histo_mask_":      tnf_batch['source_mask'], 
        #"mri_":             tnf_batch['target_image'], 
        #"mri_mask_":        tnf_batch['target_mask']
    }
    for item in save_images:
        save_results(item + name, save_images[item])
            
    # Calculate losses
    landmark_loss        = torch.sqrt(torch.mean((warped_landmarks - tnf_batch['target_landmarks'])*(warped_landmarks - tnf_batch['target_landmarks']))).data.cpu().numpy()
    total_landmark_loss  += landmark_loss
    print('\n***** Sample ' + name + ' *****')
    print('Landmark SSD loss: '+ str(landmark_loss))
    
    dice_loss        = calc_dice(tnf_batch['source_mask'][:,0,:,:], tnf_batch['target_mask'][:,0,:,:])
    total_dice_loss += dice_loss
    
total_landmark_loss /= len(dataloader_test)
total_dice_loss     /= len(dataloader_test)
print('\nAverage landmark SSD loss: {:.6f}'.format(total_landmark_loss))
print('Average DICE loss: {:.6f}'.format(total_dice_loss))
