# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:13:02 2021

@author: wyzzh
"""
import os
from skimage import io
from skimage.morphology import disk, erosion, dilation, white_tophat, reconstruction
from skimage.measure import label, regionprops
import numpy as np
from PIL import Image
import pandas as pd

tophat_disk_size = 10
bg_thres = 20
erode_size = 2
cut_margin = True 
margin_size = 5

imgFile = r"C:\Users\wyzzh\Desktop\Klenerman Lab\Scripts\GUI\Rons test\samples_results\samples\X0Y0R1W1C0_488.tif"
img = io.imread(imgFile) # Read image

if len(img.shape)==3: # Determine if the image is a stack file with multiple slices
    img = np.mean(img, axis=0) # If true, average the image
else:
    pass # If already averaged, go on processing

tophat_disk = disk(tophat_disk_size) # create tophat structural element disk, diam = tophat_disk_size (typically set to 10)
tophat_img = white_tophat(img, tophat_disk) # Filter image with tophat
top_img_subbg = tophat_img 
top_img_subbg[tophat_img<bg_thres]=0 # clean out array elements smaller than bg_thres, usually set to 40
binary_img = top_img_subbg 
binary_img[binary_img>0]=1 # binarise the image, non-positive elements will be set as 1
erode_disk = disk(erode_size) # create erode structural element disk, diam = erode_size (typically set to 2)
erode_img = erosion(top_img_subbg, erode_disk) # erode image, remove small dots (possibly noise)
dilate_img = dilation(erode_img, erode_disk) # dilate image, recover eroded elements

if cut_margin is True:
    margin = np.ones(np.shape(img))
    margin[0:margin_size, :] = 0
    margin[-margin_size:-1, :] = 0
    margin[:, 0:margin_size] = 0
    margin[:, -margin_size:-1] = 0
    mask = dilate_img*margin
else:
    mask = dilate_img
masked_img = mask*img # return masked image

inverse_mask = 1-mask
img_bgonly = inverse_mask*img
seed_img = np.copy(img_bgonly) #https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_holes_and_peaks.html
seed_img[1:-1, 1:-1] = img_bgonly.max()
seed_mask = img_bgonly
filled_img = reconstruction(seed_img, seed_mask, method='erosion')
img_nobg = abs(img - filled_img)

# Label the image to index all aggregates
labeled_img = label(mask)
area_list = []
centroid_x_list = []
centroid_y_list = []
x_min_list= []
y_min_list= []
x_max_list= []
y_max_list= []
intensity_list = []
avg_intensity_list = []
Abs_frame = []
Channel = []
Slice = []
Frame = []

# Get the number of particles
num_aggregates = int(np.max(labeled_img))
# Get profiles of labeled image
region_props = regionprops(labeled_img)

# Analyze each particle
for j in range(0, num_aggregates):
    area_aggregate = region_props[j].area
    area_list.append(area_aggregate)
    
    centroid_x, centroid_y = region_props[j].centroid
    centroid_x_list.append(centroid_x)
    centroid_y_list.append(centroid_y)
    
    x_min, y_min, x_max, y_max = region_props[j].bbox
    x_min_list.append(x_min)
    y_min_list.append(y_min)
    x_max_list.append(x_max)
    y_max_list.append(y_max)
    
    
    current_aggregate = np.copy(labeled_img)
    current_aggregate[current_aggregate!=j+1]=0
    current_aggregate[current_aggregate>0] =1
    intensity = np.sum(current_aggregate*img_nobg)
    intensity_list.append(intensity)
    
    Abs_frame.append(1)
    Channel.append(1)
    Slice.append(1)
    Frame.append(1)
#%%
area_array = np.asarray(area_list)
intensity_array = np.asarray(intensity_list)
centroid_x_array = np.asarray(centroid_x_list)
centroid_y_array = np.asarray(centroid_y_list)
x_min_array = np.asarray(x_min_list)
y_min_array = np.asarray(y_min_list)
x_max_array = np.asarray(x_max_list)
y_max_array = np.asarray(y_max_list)
Abs_frame_array = np.asarray(Abs_frame)
Channel_array = np.asarray(Channel)
Slice_array = np.asarray(Slice)
Frame_array = np.asarray(Frame)

area_array = np.reshape(area_array,[num_aggregates, 1])
intensity_array = np.reshape(intensity_array,[num_aggregates, 1])
centroid_x_array = np.reshape(centroid_x_array,[num_aggregates, 1])
centroid_y_array = np.reshape(centroid_y_array,[num_aggregates, 1])
x_min_array = np.reshape(x_min_array,[num_aggregates, 1])
y_min_array = np.reshape(y_min_array,[num_aggregates, 1])
x_max_array = np.reshape(x_max_array,[num_aggregates, 1])
y_max_array = np.reshape(y_max_array,[num_aggregates, 1])
Abs_frame_array = np.reshape(Abs_frame_array,[num_aggregates, 1])
Channel_array = np.reshape(Channel_array,[num_aggregates, 1])
Slice_array = np.reshape(Slice_array,[num_aggregates, 1])
Frame_array = np.reshape(Frame_array,[num_aggregates, 1])

result = np.concatenate((Abs_frame_array, centroid_x_array, centroid_y_array, 
                         Channel_array, Slice_array, Frame_array, x_min_array, 
                         y_min_array, x_max_array, y_max_array, area_array, 
                         intensity_array), axis = 1)

result_csv = pd.DataFrame(result)
result_csv.columns = ['Abs_frame', 'X_(px)', 'Y_(px)', 'Channel', 'Slice', 
                      'Frame', 'xMin', 'yMin', 'xMax', 'yMax', 'NArea', 
                      'IntegrateInt']
