import numpy as np
import cv2 
import SimpleITK as sitk 
from collections import OrderedDict
from landmark_functions import *
import os 


def make_dir(dir):
    try: 
        os.mkdir(dir)  
    except: 
        pass 
    
def transformAndSaveRegion(preprocess_moving_dest, case, slice, s, region, theta, dH, dW, h, w, x, y, x_offset, y_offset): 
    rotated = np.zeros((dH, dW, 3))   
    try:
        path = s['regions'][region]['filename']
        ann  = cv2.imread(path) #annotation
        ann  = np.pad(ann,((ann.shape[0],ann.shape[0]),(ann.shape[1],ann.shape[1]),(0,0)),'constant', constant_values=0)
        
        # Rotate
        rows, cols, _   = ann.shape
        M               = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
        rotated_ann     = cv2.warpAffine(ann,M,(cols,rows))

        try: 
            # flip image vertically
            if s['transform']['flip_v'] == 1: 
                rotated_ann = cv2.flip(rotated_ann, 0)
            # flip image horizontally
            if s['transform']['flip_h'] == 1: 
                rotated_ann = cv2.flip(rotated_ann, 1)
        except: 
            pass 
        
        if region != 'density':
            # find edge and downsample
            rotated_ann[rotated_ann > 0] = 1

            # set edge to outline 
            padRegion = np.zeros((w + 2*x_offset, h + 2*y_offset, 3))
            padRegion[x_offset:w + x_offset, y_offset:h + y_offset,:] = (rotated_ann[x:x+w,y:y+h]>0)*255 
        else:
            # set edge to outline 
            padRegion = np.zeros((w + 2*x_offset, h + 2*y_offset))
            padRegion[x_offset:w + x_offset, y_offset:h + y_offset] = rotated_ann[x:x+w,y:y+h,0]
        
        rotated = cv2.resize(padRegion, (dH, dW), interpolation=cv2.INTER_CUBIC)
    except: 
        pass
    
    make_dir(preprocess_moving_dest + case)
    
    outputPath = preprocess_moving_dest + case + '/' + region + '_' + case + '_' + slice +'.png'
    # ex. region = mask, case = aaa0069, slice = slice1
    
    cv2.imwrite(outputPath, rotated)


# preprocess_hist into hist slices here
def preprocess_hist(moving_dict, pre_process_moving_dest, case, dwi=False, fIC=''): 
    landmarks   = {}
    count       = 0
    for slice in moving_dict:
        s = moving_dict[slice]
        
        # Read image
        img  = cv2.imread(s['filename'], )
        row, col, _ = img.shape

        # multiply by mask
        mask    = cv2.imread(s['regions']['mask']['filename'])
        img     = img*(mask/255)

        # find rotation
        try: 
            theta = -s['transform']['rotation_angle']
        except: 
            theta = 0
    
        img         = np.pad(img,((img.shape[0],img.shape[0]),(img.shape[1],img.shape[1]),(0,0)),'constant', constant_values=0)
        mask        = np.pad(mask,((mask.shape[0],mask.shape[0]),(mask.shape[1],mask.shape[1]),(0,0)),'constant', constant_values=0)
        
        # Rotate image
        rows, cols, _   = img.shape
        M               = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
        rotated_hist    = cv2.warpAffine(img,M,(cols,rows),borderValue = (0,0,0))
        rotated_mask    = cv2.warpAffine(mask,M,(cols,rows))
        
        try: 
            # flip image vertically
            if s['transform']['flip_v'] == 1: 
                rotated_hist = cv2.flip(rotated_hist, 0)
                rotated_mask = cv2.flip(rotated_mask, 0)
            # flip image horizontally
            if s['transform']['flip_h'] == 1: 
                rotated_hist = cv2.flip(rotated_hist, 1)
                rotated_mask = cv2.flip(rotated_mask, 1)
        except: 
            pass 

        # use mask to create a bounding box around slice
        points      = np.argwhere(rotated_mask[:,:,0] != 0)
        points      = np.fliplr(points)                      # store in x,y coordinates instead of row,col indices
        y, x, h, w  = cv2.boundingRect(points)           # create a rectangle around those points
        
        crop = rotated_hist[x:x+w, y:y+h,:]
        if h>w:
            y_offset = int(h*0.15)
            x_offset = int((h - w + 2*y_offset)/2)
        else:
            y_offset = int(h*0.2)
            x_offset = int((h - w + 2*y_offset)/2)
            
        # pad image
        padHist = np.zeros((w + 2*x_offset, h + 2*y_offset, 3)) 
        padHist[x_offset:crop.shape[0]+x_offset, y_offset:crop.shape[1]+y_offset, :] = crop

        # downsample image
        size_high = 1024
        padHist_high_res    = cv2.resize(padHist, (size_high, size_high), interpolation=cv2.INTER_CUBIC)
        
        ### Keep???? 
        if dwi:
            padHist         = cv2.resize(padHist, (100, 100), interpolation=cv2.INTER_CUBIC)
        size_low  = 240
        padHist_low_res     = cv2.resize(padHist, (size_low, size_low), interpolation=cv2.INTER_CUBIC)
        
        # Transform all regions
        for region in s['regions']:
            transformAndSaveRegion(pre_process_moving_dest, case, slice,               s, region, theta, size_low,  size_low,  h, w, x, y, x_offset, y_offset)
            transformAndSaveRegion(pre_process_moving_dest, case + '_high_res', slice, s, region, theta, size_high, size_high, h, w, x, y, x_offset, y_offset)

        # Write images, with new filename
        cv2.imwrite(pre_process_moving_dest + case + '/hist_' + case + '_' + slice +'.png', padHist_low_res)
        cv2.imwrite(pre_process_moving_dest + case + '_high_res' + '/hist_' + case + '_high_res_' + slice +'.png', padHist_high_res)

        """
        try:
            dims                = x, y, w, h, x_offset, y_offset
            landmarks[count], landmarks['size-'+str(count)] = preprocess_landmarks(s, row, col, M, fIC, dims, 240)

            #for i in range(8): # NOT 8, NUM LANDMARKS!
            #    cv2.imwrite(pre_process_moving_dest + case + '/' + case + '_landmark_'+str(i)+'_slice_' + slice +'.png', landmarks[count][:,:,i]*255)
            count              += 1
            
        except:
            pass
        """
        
    return landmarks
        

def getArray(img_path):
    img     = sitk.ReadImage(img_path)
    array   = sitk.GetArrayFromImage(img)
    return array
    
#preprocess mri mha files to slices here
def preprocess_mri(fixed_img_mha, fixed_seg, pre_process_fixed_dest, coord, case, dwi_map='', fIC=False, cancer=None, healthy=None, landmarks=None, exvivo=False, target=True, fIC_slice=None, reg_fIC=False):  
    make_dir(pre_process_fixed_dest + case)

    imMri = getArray(fixed_img_mha)
    if len(imMri.shape) > 3:
        imMri = np.squeeze(imMri[0,:,:,:])
        
    imMriMask   = sitk.ReadImage(fixed_seg)
    maskArray   = sitk.GetArrayFromImage(imMriMask)
    
    if dwi_map:
        im_dwi_map  = getArray(dwi_map)
    
    if cancer:
        im_cancer   = getArray(cancer)
        
    if healthy:
        im_healthy  = getArray(healthy)
    
    if landmarks:
        make_dir(pre_process_fixed_dest + case + '/landmarks/')

        landmark_list =  [pos for pos in sorted(os.listdir(landmarks))]
        landmark_imgs = {}
        for i in range(len(landmark_list)):
            landmark_imgs[i] = getArray(landmarks + landmark_list[i])
    
    #### resample mri mask to be the same size as mri
    if (imMri.shape[1]!=maskArray.shape[1] or imMri.shape[2]!=maskArray.shape[2]):
        print("Input MRI and MRI mask have different sizes. Reshaping mask.")
        print(imMri.shape, maskArray.shape)
        mri_ori     = sitk.ReadImage(fixed_img_mha)
        resampler   = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(mri_ori)
        imMriMask   = resampler.Execute(imMriMask)
    
    imMriMask = sitk.GetArrayFromImage(imMriMask)
    
    coord[case] = {}
    coord[case]['x_offset'] = []
    coord[case]['y_offset'] = []
    coord[case]['x'] = []
    coord[case]['y'] = []
    coord[case]['h'] = []
    coord[case]['w'] = []
    coord[case]['slice']  = []
        
    for slice in range(imMri.shape[0]):
        if np.sum(np.ndarray.flatten(imMriMask[slice, :, :])) == 0: 
            continue
        
        mri_mask = imMriMask[slice, :, :] 
        if np.amax(mri_mask) == 1:
            mri_mask *= 255
        
        # create a bounding box around slice
        points = np.argwhere(mri_mask != 0)
        points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
        y, x, h, w = cv2.boundingRect(points) # create a rectangle around those points
        
        if not reg_fIC:
            # if registering histo to fIC, do not normalise fIC! 
            imMri[slice, :, :] = imMri[slice, :, :] / np.max(imMri[slice, :, :]) 
        imMri[slice, :, :] *= 255
        mri = imMri[slice, :, :] * imMriMask[slice, :, :]
   
        if exvivo:
            mri         = cv2.flip(mri, 1)
            mri_mask    = cv2.flip(mri_mask, 1)
            
        if h>w:
            y_offset = int(h*0.15)
            x_offset = int((h - w + 2*y_offset)/2)
        else:
            y_offset = int(h*0.2)
            x_offset = int((h - w + 2*y_offset)/2)
        
        # save x, y, x_offset, y_offset, h, w for each slice in dictionary 'coord' (coordinates)
        coord[case]['x'].append(x)
        coord[case]['y'].append(y)
        coord[case]['h'].append(h)
        coord[case]['w'].append(w)
        coord[case]['slice'].append(slice) 
        coord[case]['x_offset'].append(x_offset)
        coord[case]['y_offset'].append(y_offset)  
        
        if x - x_offset < 0:
            min_x = 0
        else:
            min_x = x - x_offset
            
        if y - y_offset < 0:
            min_y = 0
        else:
            min_y = y - y_offset
        
        cropMri     = mri[min_x:x+w+x_offset, min_y:y+h +y_offset]
        cropMri     = cropMri*25.5/(np.max(cropMri)/10)
        cropMask    = mri_mask[min_x:x+w+x_offset, min_y:y+h +y_offset]
        
        new_h = int(h + 2*y_offset)
        new_w = int(w + 2*x_offset)
        
        upsMri  = cv2.resize(cropMri.astype('float32'), (new_h,  new_w), interpolation=cv2.INTER_CUBIC)
        upsMask = cv2.resize(cropMask.astype('float32'), (new_h,  new_w), interpolation=cv2.INTER_CUBIC)
            
        # write to a file        
        cv2.imwrite(pre_process_fixed_dest + case + '/mri_' + case + '_' + str(slice).zfill(2) +'.jpg', upsMri)  
        cv2.imwrite(pre_process_fixed_dest + case + '/mri_uncropped_' + case + '_' + str(slice).zfill(2) +'.jpg', imMri[slice, :, :])
        if target:
            cv2.imwrite(pre_process_fixed_dest + case + '/mri_mask_' + case + '_' + str(slice).zfill(2) +'.jpg', np.uint8(mri_mask))
        else:
            cv2.imwrite(pre_process_fixed_dest + case + '/mri_mask_' + case + '_' + str(slice).zfill(2) +'.jpg', np.uint8(upsMask))
            
        if dwi_map:
            if fIC_slice == None or fIC_slice == '':
                map_slice = slice
            else:
                map_slice = fIC_slice
            
            if fIC:
                ####### NOTE: we are scaling the pixel values ####### 
                im_dwi_map_slice = im_dwi_map[int(map_slice), :, :] / 2 * 255
                tag = '/fIC_'
            else:
                im_dwi_map_slice = im_dwi_map[int(map_slice), :, :] / 3 * 255
                tag = '/ADC_'
            
            cv2.imwrite(pre_process_fixed_dest + case + tag + case + '_' + str(slice).zfill(2) +'.jpg', np.uint8(im_dwi_map_slice))
            
        if cancer:
            cancer_slice = im_cancer[slice, :,:]
            if np.amax(cancer_slice) == 1:
                cancer_slice *= 255
            ups_cancer   = cv2.resize(cancer_slice[min_x:x+w+x_offset, min_y:y+h +y_offset].astype('float32'), (new_h,  new_w), interpolation=cv2.INTER_CUBIC)
            
            if target:
                cv2.imwrite(pre_process_fixed_dest + case + '/cancer_' + case + '_' + str(slice).zfill(2) +'.jpg', np.uint8(cancer_slice))
            else:
                cv2.imwrite(pre_process_fixed_dest + case + '/cancer_' + case + '_' + str(slice).zfill(2) +'.jpg', np.uint8(ups_cancer))
        
        if healthy:
            healthy_slice = im_healthy[slice, :,:]
            if np.amax(healthy_slice) == 1:
                healthy_slice *= 255
            ups_healthy   = cv2.resize(healthy_slice[min_x:x+w+x_offset, min_y:y+h +y_offset].astype('float32'), (new_h,  new_w), interpolation=cv2.INTER_CUBIC)
            
            if target:
                cv2.imwrite(pre_process_fixed_dest + case + '/healthy_' + case + '_' + str(slice).zfill(2) +'.jpg', np.uint8(healthy_slice))
            else:
                cv2.imwrite(pre_process_fixed_dest + case + '/healthy_' + case + '_' + str(slice).zfill(2) +'.jpg', np.uint8(ups_healthy))
            
        if landmarks:
            dims = (min_x,x,w,x_offset,min_y,y,h,y_offset, h, w)
            preprocess_mri_landmarks(landmark_imgs, dims, case, slice, pre_process_fixed_dest + case + '/landmarks/')
            
        
    coord = OrderedDict(coord)
    
    return coord