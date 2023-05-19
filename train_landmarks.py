  
"""
If you use this code, please cite the following papers:
(1) Shao, Wei, et al. "ProsRegNet: A Deep Learning Framework for Registration of MRI and Histopathology Images of the Prostate." Medical Image Analysis. 2020.
(2) Rocco, Ignacio, Relja Arandjelovic, and Josef Sivic. "Convolutional neural network architecture for geometric matching." Proceedings of CVPR. 2017.

The following code is adapted from: https://github.com/ignacio-rocco/cnngeometric_pytorch.
"""

from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn     as nn
import torch.optim  as optim
import numpy        as np
import pandas       as pd
from torch.utils.data           import Dataset, DataLoader
from model.ProsRegNet_model     import ProsRegNet
from image.normalization        import NormalizeImageDict
from util.torch_util            import save_checkpoint
from util.train_test_fn         import test
from data.synth_dataset         import SynthDataset
from data.landmark_dataset      import LandmarkDataset
from geotnf.transformation      import SynthPairTnf
from collections                import OrderedDict
from torch.optim.lr_scheduler   import StepLR
from collections                import OrderedDict
from geotnf.transformation      import GeometricTnf
from skimage                    import io
from torch.autograd             import Variable

# Ignore warnings
import warnings
warnings.simplefilter("ignore", UserWarning)

class LandmarkTnf(object):

    def __init__(self, use_cuda=True, crop_factor=16/16, output_size=(240,240), padding_factor = 0.0):
        assert isinstance(use_cuda, (bool))
        assert isinstance(crop_factor, (float))
        assert isinstance(output_size, (tuple))
        assert isinstance(padding_factor, (float))
        self.use_cuda           = use_cuda
        self.crop_factor        = crop_factor
        self.padding_factor     = padding_factor
        self.out_h, self.out_w  = output_size 
        self.rescalingTnf = GeometricTnf('affine', self.out_h, self.out_w, use_cuda = self.use_cuda)
        
    def __call__(self, batch):
        image_batch_A, image_batch_B = batch['source_image'], batch['target_image']
        
        if self.use_cuda:
            image_batch_A = image_batch_A.cuda()
            image_batch_B = image_batch_B.cuda()
                          
        # generate symmetrically padded image for bigger sampling region
        #image_batch_A = self.symmetricImagePad(image_batch_A,self.padding_factor)
        #image_batch_B = self.symmetricImagePad(image_batch_B,self.padding_factor)
        
        # convert to variables
        histo_image_batch   = Variable(image_batch_A,requires_grad=False)
        mri_image_batch     = Variable(image_batch_B,requires_grad=False)

        # get cropped image
        #cropped_image_batch = self.rescalingTnf(image_batch_A,None,self.padding_factor,self.crop_factor) # Identity is used as no theta given
        
        Ones_A  = torch.ones(histo_image_batch.size())
        Zeros_A = torch.zeros(histo_image_batch.size())
        Ones_B  = torch.ones(mri_image_batch.size())
        Zeros_B = torch.zeros(mri_image_batch.size())
        
        if self.use_cuda:
            Ones_A  = Ones_A.cuda()
            Zeros_A = Zeros_A.cuda()
            Ones_B  = Ones_B.cuda()
            Zeros_B = Zeros_B.cuda()
            
        histo_mask_batch = torch.where(histo_image_batch > 0.1*Ones_A, Ones_A, Zeros_A)
        mri_mask_batch   = torch.where(mri_image_batch   > 0.1*Ones_B, Ones_B, Zeros_B)
        
        if self.use_cuda:
            histo_image_batch   = histo_image_batch.cuda()
            mri_image_batch     = mri_image_batch.cuda()
            histo_mask_batch    = histo_mask_batch.cuda()
            mri_mask_batch      = mri_mask_batch.cuda()
        
        #mask1 = 255*normalize_image(warped_mask_batch,forward=False)
        #mask1 = mask1.data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()
        
        #print(mask1.shape)

        #io.imsave('warped_mask.jpg', mask1)

        return {'source_image': histo_image_batch, 'target_image': mri_image_batch, 'source_mask': histo_mask_batch, 'target_mask': mri_mask_batch}


def train(epoch, model, loss_fn, optimizer, dataloader, landmark_tnf, use_cuda=True):
    model.train()
    train_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        tnf_batch   = landmark_tnf(batch)
        theta       = model(tnf_batch)
        
        # Apply transformation to landmarks
        tps_tnf          = GeometricTnf('tps', use_cuda=use_cuda)
        warped_landmarks = tps_tnf(batch['landmarks_source'], theta)
        
        # Warped landmarks is a (h x w x num_landmarks)
        _,_,count       = warped_landmarks.shape
        landmark_list   = np.zeros((2, count))
        
        for i in range(count):
            (x,y)              = np.unravel_index(np.argmax(warped_landmarks[:,:,i], axis=None), warped_landmarks[:,:,i].shape)  
            landmark_list[:,i] = [x,y]

        loss = loss_fn(landmark_list,batch['landmarks_target'])
        
        loss.backward()
        optimizer.step()
        train_loss += loss.data.cpu().numpy()
        
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
            epoch, batch_idx , len(dataloader),
            100. * batch_idx / len(dataloader), loss.data))
        
    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.6f}'.format(train_loss))
    return train_loss


# Argument parsing
parser = argparse.ArgumentParser(description='ProsRegNet PyTorch implementation')

# Paths
parser.add_argument('-t', '--training-csv-name',   type=str, default='train_landmarks.csv', help='training transformation csv file name')
parser.add_argument(      '--test-csv-name',       type=str, default='test.csv',            help='test transformation csv file name')

parser.add_argument('-p', '--training-image-path', type=str, default='datasets/training_landmarks/',  help='path to folder containing training images')
parser.add_argument(      '--trained-models-dir',  type=str, default='trained_models',                help='path to trained models folder')
parser.add_argument('-n', '--trained-models-name', type=str, default='default',               help='trained model filename')

# Optimization parameters 
parser.add_argument('--lr',             type=float, default=0.0003, help='learning rate')
parser.add_argument('--gamma',          type=float, default=0.95,   help='gamma')
parser.add_argument('--momentum',       type=float, default=0.9,    help='momentum constant')
parser.add_argument('--num-epochs',     type=int,   default=10,     help='number of training epochs')
parser.add_argument('--batch-size',     type=int,   default=4,     help='training batch size')
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

model       = ProsRegNet(use_cuda=use_cuda,geometric_model='tps',feature_extraction_cnn='resnet101')
model_path  = os.path.join(args.trained_models_dir, 'best_' + args.trained_models_name + '_tps.pth.tar')
checkpoint  = torch.load(model_path, map_location=lambda storage, loc: storage)
checkpoint['state_dict'] = OrderedDict([(k.replace('resnet101', 'model'), v) for k, v in checkpoint['state_dict'].items()])
model.load_state_dict(checkpoint['state_dict'])

print('Using MSE loss...')
loss = nn.MSELoss()

# Dataset and dataloader
dataset_train = LandmarkDataset(geometric_model  = 'tps',
                            csv_file             = './training_data/tps/' + args.training_csv_name,
                            training_image_path  = args.training_image_path
                            #transform            = NormalizeImageDict(['Histo','MRI'])
                            )

dataset_test = SynthDataset(geometric_model      = 'tps',
                            csv_file             = './training_data/tps/' + args.test_csv_name,
                            training_image_path  = args.training_image_path,
                            transform            = NormalizeImageDict(['image_A','image_B']))

dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
dataloader_test  = DataLoader(dataset_test,  batch_size=args.batch_size, shuffle=True, num_workers=4)


pair_generation_tnf = SynthPairTnf(geometric_model='tps',use_cuda=use_cuda)
landmark_tnf        = LandmarkTnf(use_cuda=use_cuda)

# Optimizer
optimizer = optim.Adam(model.FeatureRegression.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

# Train
checkpoint_name = os.path.join(args.trained_models_dir, args.trained_models_name + '_tps_landmarks.pth.tar')
best_test_loss  = float("inf")

print('Starting training...')

epochArray      = np.zeros(args.num_epochs)
trainLossArray  = np.zeros(args.num_epochs)
testLossArray   = np.zeros(args.num_epochs)

for epoch in range(1, args.num_epochs+1):
    train_loss = train(epoch,model,loss,optimizer,dataloader_train,landmark_tnf)
    test_loss  = test(model,loss,dataloader_test,pair_generation_tnf,use_cuda=use_cuda, geometric_model='tps')
    
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
np.savetxt(os.path.join(args.trained_models_dir, args.trained_models_name + '_tps_landmarks.csv'), np.transpose((epochArray, trainLossArray, testLossArray)), delimiter=',')
