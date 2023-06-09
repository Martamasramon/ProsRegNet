  
"""
If you use this code, please cite the following papers:
(1) Shao, Wei, et al. "ProsRegNet: A Deep Learning Framework for Registration of MRI and Histopathology Images of the Prostate." Medical Image Analysis. 2020.
(2) Rocco, Ignacio, Relja Arandjelovic, and Josef Sivic. "Convolutional neural network architecture for geometric matching." Proceedings of CVPR. 2017.

The following code is adapted from: https://github.com/ignacio-rocco/cnngeometric_pytorch.
"""

from __future__ import print_function, division
import argparse
import os
import wandb
import torch
import torch.nn     as nn
import torch.optim  as optim
import numpy        as np
from torch.utils.data           import DataLoader
from model.ProsRegNet_model     import ProsRegNet
from model.loss                 import SSDLoss
from data.synth_dataset         import SynthDataset
from geotnf.transformation      import SynthPairTnf
from image.normalization        import NormalizeImageDict
from util.train_test_fn         import train, test
from util.torch_util            import save_checkpoint, str_to_bool
from collections                import OrderedDict
from torch.optim.lr_scheduler   import StepLR

# Ignore warnings
import warnings
warnings.simplefilter("ignore", UserWarning)

# Argument parsing
parser = argparse.ArgumentParser(description='ProsRegNet PyTorch implementation')

# Paths
parser.add_argument('-t', '--training-csv-name',   type=str, default='train.csv',      help='training transformation csv file name')
parser.add_argument('-s', '--test-csv-name',       type=str, default='test.csv',       help='test transformation csv file name')

parser.add_argument('-p', '--training-image-path', type=str, default='datasets/training/',  help='path to folder containing training images')
parser.add_argument(      '--trained-models-dir',  type=str, default='trained_models',      help='path to trained models folder')
parser.add_argument('-n', '--trained-models-name', type=str, default='default',             help='trained model filename')

parser.add_argument('--pretrained-model-aff', type=str, default='', help='path to a pretrained affine network')
parser.add_argument('--pretrained-model-tps', type=str, default='', help='path to a pretrained tps network')

# Optimization parameters 
parser.add_argument('--lr',             type=float, default=0.0003, help='learning rate')
parser.add_argument('--gamma',          type=float, default=0.95,   help='gamma')
parser.add_argument('--momentum',       type=float, default=0.9,    help='momentum constant')
parser.add_argument('--num-epochs',     type=int,   default=50,     help='number of training epochs')
parser.add_argument('--batch-size',     type=int,   default=64,     help='training batch size')
parser.add_argument('--weight-decay',   type=float, default=0,      help='weight decay constant')
parser.add_argument('--seed',           type=int,   default=1,      help='Pseudo-RNG seed')

# Model parameters
parser.add_argument('--geometric-model',        type=str,                                default='affine',    help='geometric model to be regressed at output: affine or tps')
parser.add_argument('--use-mse-loss',           type=str_to_bool, nargs='?', const=True, default=False,       help='Use MSE loss on tnf. parameters')
parser.add_argument('--feature-extraction-cnn', type=str,                                default='resnet101', help='Feature extraction architecture: vgg/resnet101')

# Synthetic dataset parameters
parser.add_argument('--random-sample', type=str_to_bool, nargs='?', const=True, default=False, help='sample random transformations')

args = parser.parse_args()

# Track training
wandb.login(key="7b96e5eef65e41f6ddbbe343c84f1586ee09f565")
# start a new wandb run to track this script
wandb.init(
    project = "ProsRegNet",
    name    = args.trained_models_name,
    
    config={
    "epochs": args.num_epochs,
    }
)

use_cuda = torch.cuda.is_available()

print("Use Cuda? ", use_cuda)

torch.cuda.set_device(0)
print("cuda:", torch.cuda.current_device())

# Seed
if use_cuda:
    torch.cuda.manual_seed(args.seed)

mri = False
if args.geometric_model=='tps-mri' or args.trained_models_name=='mri':
    mri=True
    
if args.geometric_model=='affine':
    training_tnf_csv = 'training_data/affine'
else:
    training_tnf_csv = 'training_data/tps' 

# CNN model and loss
print('Creating CNN model...')

model = ProsRegNet(use_cuda=use_cuda,geometric_model=args.geometric_model,feature_extraction_cnn=args.feature_extraction_cnn, mri=mri)

if args.geometric_model == 'affine' and args.pretrained_model_aff!='':
    checkpoint = torch.load(args.pretrained_model_aff, map_location=lambda storage, loc: storage)
    checkpoint['state_dict'] = OrderedDict([(k.replace(args.feature_extraction_cnn, 'model'), v) for k, v in checkpoint['state_dict'].items()])
    model.load_state_dict(checkpoint['state_dict'])
        
if args.geometric_model == 'tps' and args.pretrained_model_tps!='':
    checkpoint = torch.load(args.pretrained_model_tps, map_location=lambda storage, loc: storage)
    checkpoint['state_dict'] = OrderedDict([(k.replace(args.feature_extraction_cnn, 'model'), v) for k, v in checkpoint['state_dict'].items()])
    model.load_state_dict(checkpoint['state_dict']) 
    
if args.geometric_model == 'tps-mri' or args.trained_models_name=='mri':
    out_size = 80
else:
    out_size = 240
    
if args.use_mse_loss:
    print('Using MSE loss...')
    loss = nn.MSELoss()
else:
    print('Using SSD loss...')
    loss = SSDLoss(use_cuda=use_cuda,geometric_model=args.geometric_model,out_size=out_size)
       
# Dataset and dataloader
dataset = SynthDataset(geometric_model      = args.geometric_model,
                       csv_file             = os.path.join(training_tnf_csv,args.training_csv_name),
                       training_image_path  = args.training_image_path,
                       output_size          = (out_size,out_size),
                       transform            = NormalizeImageDict(['image_A','image_B']),
                       random_sample        = args.random_sample)

dataset_test = SynthDataset(geometric_model     = args.geometric_model,
                            csv_file            = os.path.join(training_tnf_csv,args.test_csv_name),
                            training_image_path = args.training_image_path,
                            output_size         = (out_size,out_size),
                            transform           = NormalizeImageDict(['image_A','image_B']),
                            random_sample       = args.random_sample)

dataloader_train = DataLoader(dataset, batch_size=args.batch_size,shuffle=True, num_workers=4)
dataloader_test  = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=4)

pair_generation_tnf = SynthPairTnf(geometric_model=args.geometric_model,output_size=(out_size,out_size),use_cuda=use_cuda)

# Optimizer
optimizer = optim.Adam(model.FeatureRegression.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

# Train
checkpoint_name = os.path.join(args.trained_models_dir, args.trained_models_name + '_' + args.geometric_model + '.pth.tar')
best_test_loss  = float("inf")

print('Starting training...')

epochArray      = np.zeros(args.num_epochs)
trainLossArray  = np.zeros(args.num_epochs)
testLossArray   = np.zeros(args.num_epochs)

for epoch in range(1, args.num_epochs+1):
    print('--- Epoch ' + str(epoch) + ' ---')
    train_loss = train(model, loss, dataloader_train, pair_generation_tnf, optimizer)
    test_loss  = test(model,  loss, dataloader_test,  pair_generation_tnf,use_cuda=use_cuda, geometric_model=args.geometric_model, out_size=out_size)

    wandb.log({"train_loss":    train_loss, 
               "test_loss":     test_loss})
    print('lr', scheduler.get_lr())

    scheduler.step()
    
    epochArray[epoch-1]     = epoch
    trainLossArray[epoch-1] = train_loss
    testLossArray[epoch-1]  = test_loss

    # remember best loss
    is_best = test_loss < best_test_loss
    best_test_loss = min(test_loss, best_test_loss)
    save_checkpoint({
      'epoch':          epoch + 1,
      'args':           args,
      'state_dict':     model.state_dict(),
      'best_test_loss': best_test_loss,
      'optimizer' :     optimizer.state_dict(),
    }, is_best, checkpoint_name)
print('Done!')

# Save model as csv
np.savetxt(os.path.join(args.trained_models_dir, args.trained_models_name + '_' + args.geometric_model + '.csv'), np.transpose((epochArray, trainLossArray, testLossArray)), delimiter=',')
wandb.finish()
