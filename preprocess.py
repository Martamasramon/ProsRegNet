import numpy as np
import cv2 
import SimpleITK as sitk 
from collections import OrderedDict
import os 

def transformAndSaveRegion(preprocess_moving_dest, case, slice, s, region, theta, dH, dW, h, w,x,y,x_offset,y_offset): 
    rotated = np.zeros((w + 2*x_offset, h + 2*y_offset, 3))   
    try:
        path = s['regions'][region]['filename']
        ann  = cv2.imread(path) #annotation
        ann = np.pad(ann,((ann.shape[0],ann.shape[0]),(ann.shape[1],ann.shape[1]),(0,0)),'constant', constant_values=0)
        
        # Rotate
        rows, cols, channels = ann.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
        rotated_ann = cv2.warpAffine(ann,M,(cols,rows))

        try: 
            # flip image vertically
            if s['transform']['flip_v'] == 1: 
                rotated_ann = cv2.flip(rotated_ann, 0)
            # flip image horizontally
            if s['transform']['flip_h'] == 1: 
                rotated_ann = cv2.flip(rotated_ann, 1)
        except: 
            pass 
        
        # find edge and downsample
        #ann = cv2.resize(rotated_ann, (dH, dW), interpolation=cv2.INTER_CUBIC)
        rotated_ann[rotated_ann > 0] = 1

        # set edge to outline 
        region3d= np.zeros((w + 2*x_offset, h + 2*y_offset, 3))
        region3d[x_offset:w + x_offset, y_offset:h + y_offset,:] = (rotated_ann[x:x+w,y:y+h]>0)*255
        
        rotated = region3d
    except: 
        pass
    
    try: 
        os.mkdir(preprocess_moving_dest + case)
    except: 
        pass 
    
    outputPath = preprocess_moving_dest + case + '/' + region + '_' + case + '_' + slice +'.png'
    # ex. region = mask, case = aaa0069, slice = slice1
    
    cv2.imwrite(outputPath, rotated)


# preprocess_hist into hist slices here
def preprocess_hist(moving_dict, pre_process_moving_dest, case): 
    for slice in moving_dict:
        s = moving_dict[slice]
        
        # Read image
        img = cv2.imread(s['filename'], )

        # multiply by mask
        prosPath = s['regions']['mask']['filename']
        mask = cv2.imread(prosPath)
        img = img*(mask/255)

        # find rotation
        try: 
            theta = -s['transform']['rotation_angle']
        except: 
            theta = 0
    
        img  = np.pad(img,((img.shape[0],img.shape[0]),(img.shape[1],img.shape[1]),(0,0)),'constant', constant_values=0)
        mask = np.pad(mask,((mask.shape[0],mask.shape[0]),(mask.shape[1],mask.shape[1]),(0,0)),'constant', constant_values=0)
        
        # Rotate image
        rows, cols, channels = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
        rotated_hist = cv2.warpAffine(img,M,(cols,rows),borderValue = (0,0,0))
        rotated_mask = cv2.warpAffine(mask,M,(cols,rows))
        
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

        

        # downsample image, this has to be consistent with the size of MRI
        dH = int(rotated_hist.shape[1]/4)
        dW = int(rotated_hist.shape[0]/4)
        #rotated_hist = cv2.resize(rotated_hist, (dH, dW), interpolation=cv2.INTER_CUBIC)
        #rotated_mask = cv2.resize(rotated_mask, (dH, dW), interpolation=cv2.INTER_CUBIC)
        

        # create a bounding box around slice
        points = np.argwhere(rotated_mask[:,:,0] != 0)
        points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
        y, x, h, w = cv2.boundingRect(points) # create a rectangle around those points
        
        crop = rotated_hist[x:x+w, y:y+h,:]
        
        if h>w:
            y_offset = int(h*0.15)
            x_offset = int((h - w + 2*y_offset)/2)
        else:
            y_offset = int(h*0.2)
            x_offset = int((h - w + 2*y_offset)/2)
            
        for region in s['regions']:
            transformAndSaveRegion(pre_process_moving_dest, case, slice, s, region, theta, dH, dW, h, w,x,y,x_offset,y_offset)
        
        # pad image
        h = h + 2*y_offset
        w = w + 2*x_offset
  
        padHist = np.zeros((w, h, 3)) 
      
        padHist[x_offset:crop.shape[0]+x_offset, y_offset:crop.shape[1]+y_offset, :] = crop

        # Write images, with new filename
        cv2.imwrite(pre_process_moving_dest + case + '/hist_' + case + '_' + slice +'.png', padHist)

#preprocess mri mha files to slices here
def preprocess_mri(fixed_img_mha, fixed_seg, pre_process_fixed_dest, coord, case):     
    imMri       = sitk.ReadImage(fixed_img_mha)
    imMri       = sitk.GetArrayFromImage(imMri)
    try:
        imMri.shape[2]
    except:
        imMri = imMri.reshape(1,imMri.shape[0],imMri.shape[1])

    imMriMask   = sitk.ReadImage(fixed_seg)
    imMriMaskArray = sitk.GetArrayFromImage(imMriMask)
    try:
        imMriMaskArray.shape[2]
    except:
        imMriMaskArray = imMriMaskArray.reshape(1,imMriMaskArray.shape[0],imMriMaskArray.shape[1])
    
    
    #### resample mri mask to be the same size as mri
    if (imMri.shape[1]!=sitk.GetArrayFromImage(imMriMask).shape[1] or imMri.shape[2]!=sitk.GetArrayFromImage(imMriMask).shape[2]):
        mri_ori = sitk.ReadImage(fixed_img_mha)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(mri_ori)
        imMriMask = resampler.Execute(imMriMask)
        print("Input MRI and MRI mask have different sizes. Reshaping mask.")
    
    imMriMask = sitk.GetArrayFromImage(imMriMask)
    try:
        imMriMask.shape[2]
    except:
        imMriMask = imMriMask.reshape(1,imMriMask.shape[0],imMriMask.shape[1])
    
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
        
        imMri[slice, :, :] = imMri[slice, :, :] / np.max(imMri[slice, :, :]) * 255
        mri = imMri[slice, :, :]*imMriMask[slice, :, :]
   
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
        crop = mri[min_x:x+w+x_offset, min_y:y+h +y_offset]
        
        h = h + 2*y_offset
        w = w + 2*x_offset
        
        crop = crop*25.5/(np.max(crop)/10)
        
        # upsample slice to approx 500 px in width
        ups = 1; 
        upsHeight = int(h*ups)
        upsWidth = int(w*ups)
        
        upsMri = cv2.resize(crop.astype('float32'), (upsHeight,  upsWidth), interpolation=cv2.INTER_CUBIC)
                
        try: 
            os.mkdir(pre_process_fixed_dest + case)
        except: 
            pass 
        
        # write to a file        
        cv2.imwrite(pre_process_fixed_dest + case + '/mri_' + case + '_' + str(slice).zfill(2) +'.jpg', upsMri)  
        cv2.imwrite(pre_process_fixed_dest + case + '/mriUncropped_' + case + '_' + str(slice).zfill(2) +'.jpg', imMri[slice, :, :])
        cv2.imwrite(pre_process_fixed_dest + case + '/mriMask_' + case + '_' + str(slice).zfill(2) +'.jpg', np.uint8(mri_mask))

    coord = OrderedDict(coord)
    
    return coord