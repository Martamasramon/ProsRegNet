  
"""
If you use this code, please cite the following papers:
(1) Shao, Wei, et al. "ProsRegNet: A Deep Learning Framework for Registration of MRI and Histopathology Images of the Prostate." Medical Image Analysis. 2020.
(2) Rocco, Ignacio, Relja Arandjelovic, and Josef Sivic. "Convolutional neural network architecture for geometric matching." Proceedings of CVPR. 2017.

The following code is adapted from: https://github.com/ignacio-rocco/cnngeometric_pytorch.
"""

from __future__ import print_function, division
import argparse
import os
import random
import cv2
import torch
import torch.nn     as nn
import torch.optim  as optim
import numpy        as np
import torch.nn.parameter as Parameter
from model.loss                 import MSELoss
from torch.utils.data           import DataLoader
from torch.autograd             import Variable
from model.ProsRegNet_model     import ProsRegNet
from image.normalization        import NormalizeImageDict
from util.torch_util            import save_checkpoint
from data.landmark_dataset      import LandmarkDataset
from geotnf.transformation_landmarks import LandmarkTnf
from collections                import OrderedDict
from torch.optim.lr_scheduler   import StepLR
from collections                import OrderedDict

# Ignore warnings
import warnings
warnings.simplefilter("ignore", UserWarning)


def train(epoch, model, loss_fn, dataloader, landmark_tnf, optimizer):
    model.train()
    train_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        tnf_batch   = landmark_tnf(batch)
        theta       = model(tnf_batch)

        loss = loss_fn(theta, tnf_batch)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.data.cpu().numpy()
        
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
            epoch, batch_idx , len(dataloader),
            100. * batch_idx / len(dataloader), loss.data))
        
    train_loss /= len(dataloader)
    
    print('Train set: Average loss: {:.6f}'.format(train_loss))
    return train_loss

def test(model,loss_fn,dataloader,landmark_tnf):
    model.eval()
    test_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        tnf_batch   = landmark_tnf(batch)
        theta       = model(tnf_batch)
        
        loss = loss_fn(theta,tnf_batch)
        test_loss += loss.data.cpu().numpy()
        
    test_loss /= len(dataloader)
    
    print('Test set: Average loss: {:.6f}'.format(test_loss))
    return test_loss



# Argument parsing
parser = argparse.ArgumentParser(description='ProsRegNet PyTorch implementation')

# Paths
parser.add_argument('-t', '--training-csv-name',   type=str, default='landmarks_train.csv', help='training data csv file name')
parser.add_argument(      '--test-csv-name',       type=str, default='landmarks_test.csv',  help='test data csv file name')

parser.add_argument('-p', '--training-image-path', type=str, default='datasets/training_landmarks/',  help='path to folder containing training images')
parser.add_argument(      '--trained-models-dir',  type=str, default='trained_models',                help='path to trained models folder')
parser.add_argument('-n', '--trained-models-name', type=str, default='default',                       help='trained model filename')

# Optimization parameters 
parser.add_argument('--lr',             type=float, default=0.0003, help='learning rate')
parser.add_argument('--gamma',          type=float, default=0.95,   help='gamma')
parser.add_argument('--momentum',       type=float, default=0.9,    help='momentum constant')
parser.add_argument('--num-epochs',     type=int,   default=30,     help='number of training epochs')
parser.add_argument('--batch-size',     type=int,   default=1,      help='training batch size')
parser.add_argument('--weight-decay',   type=float, default=0,      help='weight decay constant')
parser.add_argument('--seed',           type=int,   default=1,      help='Pseudo-RNG seed')

args     = parser.parse_args()

## CUDA
use_cuda = torch.cuda.is_available() 
print("Use Cuda? ", use_cuda)
#torch.cuda.set_device(0)
#print("cuda:", torch.cuda.current_device())

# Seed
if use_cuda:
    torch.cuda.manual_seed(args.seed)

## CNN model and loss
print('Creating CNN model...')
geometric_model = 'tps'

model       = ProsRegNet(use_cuda=use_cuda,geometric_model=geometric_model,feature_extraction_cnn='resnet101')
model_path  = os.path.join(args.trained_models_dir, 'best_' + args.trained_models_name + '_tps.pth.tar')

checkpoint               = torch.load(model_path, map_location=lambda storage, loc: storage)
checkpoint['state_dict'] = OrderedDict([(k.replace('resnet101', 'model'), v) for k, v in checkpoint['state_dict'].items()])
model.load_state_dict(checkpoint['state_dict'])

print('Using MSE loss...')
mse_loss = MSELoss(use_cuda=use_cuda)

# Dataset and dataloader
dataset_train = LandmarkDataset(geometric_model  = geometric_model,
                            csv_file             = './training_data/tps/' + args.training_csv_name,
                            training_image_path  = args.training_image_path,
                            batch_size           = args.batch_size,
                            transform            = NormalizeImageDict(['source_image','target_image']))

dataset_test = LandmarkDataset(geometric_model      = geometric_model,
                            csv_file             = './training_data/tps/' + args.test_csv_name,
                            training_image_path  = args.training_image_path,
                            batch_size           = args.batch_size,
                            transform            = NormalizeImageDict(['source_image','target_image']))

dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
dataloader_test  = DataLoader(dataset_test,  batch_size=args.batch_size, shuffle=True, num_workers=4)

landmark_tnf = LandmarkTnf(use_cuda=use_cuda)

# Optimizer
optimizer = optim.Adam(model.FeatureRegression.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

# Train
checkpoint_name = os.path.join(args.trained_models_dir, args.trained_models_name + '_tps_landmarks.pth.tar')

print('Starting training...')

epochArray      = np.zeros(args.num_epochs)
trainLossArray  = np.zeros(args.num_epochs)
#testLossArray   = np.zeros(args.num_epochs)

best_loss  = float("inf")
for epoch in range(1, args.num_epochs+1):
    train_loss = train(epoch, model, mse_loss, dataloader_train, landmark_tnf, optimizer)
    #test_loss  = test(        model, mse_loss, dataloader_test,  landmark_tnf)
    
    scheduler.step()
    
    epochArray[epoch-1]     = epoch
    trainLossArray[epoch-1] = train_loss
    #testLossArray[epoch-1]  = test_loss

    # remember best loss
    is_best     = train_loss < best_loss 
    best_loss   = min(train_loss, best_loss)
    save_checkpoint({
      'epoch':          epoch + 1,
      'args':           args,
      'state_dict':     model.state_dict(),
      'best_loss':      best_loss,
      'optimizer' :     optimizer.state_dict(),
    }, is_best, checkpoint_name)
print('Done!')

# Save model as csv
np.savetxt(os.path.join(args.trained_models_dir, args.trained_models_name + '_tps_landmarks.csv'), np.transpose((epochArray, trainLossArray)), delimiter=',')
#np.savetxt(os.path.join(args.trained_models_dir, args.trained_models_name + '_tps_landmarks.csv'), np.transpose((epochArray, trainLossArray, testLossArray)), delimiter=',')
