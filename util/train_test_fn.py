"""
The following code is adapted from: https://github.com/ignacio-rocco/cnngeometric_pytorch.
"""

from __future__ import print_function, division
import torch
from geotnf.transformation import GeometricTnf
import torch

def train(model,loss_fn,dataloader,pair_generation_tnf,optimizer):
    model.train()
    train_loss = 0
    for _, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        tnf_batch   = pair_generation_tnf(batch)
        theta       = model(tnf_batch)
        
        loss        = loss_fn(theta, tnf_batch)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.data.cpu().numpy()
        
    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.6f}'.format(train_loss))
    return train_loss

def test(model,loss_fn,dataloader,pair_generation_tnf,use_cuda=True,geometric_model='affine', out_size=240):
    model.eval()
    test_loss = 0
    dice = 0
    for _, batch in enumerate(dataloader):
        tnf_batch   = pair_generation_tnf(batch)
        theta       = model(tnf_batch)
        
        loss       = loss_fn(theta,tnf_batch)
        test_loss += loss.data.cpu().numpy()
        
        target = tnf_batch['target_mask']
        geometricTnf = GeometricTnf(geometric_model, out_h=out_size, out_w=out_size, use_cuda = use_cuda)

        if geometric_model == 'affine':
            theta = theta.view(-1,2,3)
        estimate = geometricTnf(tnf_batch['source_mask'],theta)
        
        if use_cuda:
            target   = target.cuda()
            estimate = estimate.cuda()
        
        numerator   = 2 * torch.sum(torch.sum(torch.sum(target * estimate,dim=3),dim=2),dim=1)
        denominator = torch.sum(torch.sum(torch.sum(target + estimate,dim=3),dim=2),dim=1)
        dice        = dice + torch.sum(numerator/(denominator + 0.00001))/target.shape[0]

    test_loss /= len(dataloader)
    dice /=len(dataloader)
    
    print('Test set: Average loss: {:.6f}'.format(test_loss))
    print('Test set: Dice: {:.6f}'.format(dice))
    return test_loss
