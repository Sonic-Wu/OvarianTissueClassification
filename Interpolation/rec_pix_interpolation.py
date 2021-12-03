#%%
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 13:28:42 2021

@author: Xinyu Wu
"""
import envi
import numpy as np
import pandas as pd
import glob, os, sys
import cv2

# adding the parent directory to basic_utils
path_lab = r"C:\Users\ReddyLabAdmin\Dropbox\Project\Project Flash\Code"
sys.path.append(path_lab)
from basic_utils.rm_utils import align_Images

path_lab_rect = r"C:\Users\ReddyLabAdmin\Dropbox\Project\Project Flash\Data\csv_rect_pixels"
path_home_rect = r"D:\Dropbox\Project\Project Flash\Data\csv_rect_pixels"

path_lab_ref = r"C:\Users\ReddyLabAdmin\Dropbox\Project\Project Flash\Data\0.5_cerv_cores"
path_home_ref = r"D:\Dropbox\Project\Project Flash\Data\0.5_cerv_cores"

save_path_lab = r"C:\Users\ReddyLabAdmin\Dropbox\Project\Project Flash\Data"
save_path_home = r"D:\Dropbox\Project\Project Flash\Data"

# setting work path
path_rect = path_lab_rect
path_ref = path_lab_ref
#%% transfer csv data to envi file
# read high resolution reference image envi file
sample_list_ref = os.listdir(path_ref)

# read high resolution rectangular piexl envi file
sample_list_rect = os.listdir(path_rect)

for sample_rect_number in range(len(sample_list_rect)):
    
    sample_rect_full_name = sample_list_rect[sample_rect_number]
    sample_rect_name = sample_rect_full_name[:sample_rect_full_name.find('_')] # using find to find "_" character to extract sample name

    try:
        index = sample_list_ref.index(sample_rect_name)
    except ValueError:
        print(f"{sample_rect_full_name}" + " doesn't have corresponding reference image")
        continue
    
    sample_ref_name = sample_list_ref[index]
    # check if we have correct paired rect and ref image
    if sample_rect_name == sample_ref_name:
        pass
    else:
        print(f"the processing image name is {sample_rect_name}, while reference image is {sample_ref_name}" )

    # reading high resolution rectangular image envi file and reference image envi file 
    envi_rect_data_path = path_rect + '\\' + sample_rect_full_name + '\\envi' + sample_rect_full_name
    envi_ref_data_path = path_ref + '\\' + sample_ref_name + '\\Envi' + sample_ref_name
    envi_rect = envi.envi(envi_rect_data_path)
    envi_ref = envi.envi(envi_ref_data_path)

    rect_multi_channel_image = envi_rect.loadall()
    rect_wavenumber = np.asarray(envi_rect.header.wavelength)

    ref_multi_channel_image = envi_ref.loadall()
    
    # interpolating on rectangular image, size based on reference image
    X_scale = ref_multi_channel_image[0].shape[1] / rect_multi_channel_image[0].shape[1]
    Y_scale = ref_multi_channel_image[0].shape[0] / rect_multi_channel_image[0].shape[0]

    for each_channel in range(rect_multi_channel_image.shape[0]):
        image_each_channel = rect_multi_channel_image[each_channel]    
        image_interp_same_size_as_ref = cv2.resize(image_each_channel, None, fx = X_scale, fy = Y_scale, interpolation = cv2.INTER_LINEAR)
        image_interp_0_5_pixel = cv2.resize(image_each_channel, None, fx = 1, fy = 10, interpolation = cv2.INTER_LINEAR)

        image_ref_size = np.reshape(image_interp_same_size_as_ref,(1,image_interp_same_size_as_ref.shape[0], image_interp_same_size_as_ref.shape[1])) 
        image_0_5_size = np.reshape(image_interp_0_5_pixel, (1, image_interp_0_5_pixel.shape[0], image_interp_0_5_pixel.shape[1]))
        image_aligned = align_Images(image_interp_same_size_as_ref, ref_multi_channel_image[0])
        image_aligned = np.reshape(image_aligned[0],(1, image_aligned[0].shape[0], image_aligned[0].shape[1]))
        
        if each_channel == 0:
            rect_image_interp_ref_size = image_ref_size
            rect_image_0_5_pixel_size = image_0_5_size
            rect_image_aligned = image_aligned
        else:
            rect_image_interp_ref_size = np.concatenate((rect_image_interp_ref_size, image_ref_size), axis = 0)
            rect_image_0_5_pixel_size = np.concatenate((rect_image_0_5_pixel_size, image_0_5_size), axis = 0)
            rect_image_aligned = np.concatenate((rect_image_aligned, image_aligned), axis = 0)
    
    # save as envi file at different folder under Data
    save_path = path_lab_rect 
    fname_0_5_pixel = save_path + '\\' + sample_rect_full_name + '\\' + 'Envi' + sample_rect_name + '_0_5_pixel'
    fname_aligned = save_path + '\\'  + sample_rect_full_name + '\\' + 'Envi' + sample_rect_name + '_aligned'
    envi.save_envi(rect_image_0_5_pixel_size, fname_0_5_pixel,wavelength = rect_wavenumber)
    envi.save_envi(rect_image_aligned, fname_aligned, wavelength = rect_wavenumber)

# -*- coding: utf-8 -*-   
##%%
## interpolation
#temp_image = rect_D[0]
#
#
#xx = [0, 5, 10]
#yy = [0, 10, 5]
#
#x = np.linspace(0, 10, 11)
#y = np.interp(x, xx, yy)
#
#test_image = pd.DataFrame(temp_image)
#image_interp = np.zeros((1200,1))
#
#for column in range(test_image.shape[1]):
#    pixel_value = test_image.loc[:,[column]]
#    
#    pixel_number = np.linspace(pixel_value.index.start, (pixel_value.index.stop-1) * 5, pixel_value.index.stop)
#    
#    pixel_number_interp = np.linspace(pixel_value.index.start, (pixel_value.index.stop-1) * 5, (pixel_value.index.stop-1) * 5)
#       
#    pixel_value_interp = np.interp(pixel_number_interp, pixel_number, np.reshape(pixel_value.values.tolist(),241))
#    
#    pixel_value_interp = pixel_value_interp.reshape((len(pixel_value_interp), 1))
#    
#    image_interp = np.concatenate([image_interp, pixel_value_interp], axis = 1)
#    
#image_interp = np.delete(image_interp,0,1)
#
#
##%% interpolate using opencv
#import cv2
#import matplotlib.pyplot as plt
#
#temp_interp_multi_channel_image = np.zeros([1,2410,2471])
#for each_band in range(len(rect_W)):
#    temp_image = rect_D[each_band]
#    bilinear_image = cv2.resize(temp_image, None, fx = 1, fy = 10, interpolation = cv2.INTER_LINEAR)
#    cubic_image = cv2.resize(temp_image, None, fx = 1, fy = 10, interpolation = cv2.INTER_CUBIC)
#    #temp_interp_multi_channel_image = np.append(temp_interp_multi_channel_image, cubic_image, axis = 0)
#    temp_interp_multi_channel_image = np.vstack([temp_interp_multi_channel_image, cubic_image.reshape((1,cubic_image.shape[0],cubic_image.shape[1]))])
#    #temp_interp_multi_channel_image = np.concatenate([temp_interp_multi_channel_image, cubic_image], axis = 0)
#
#temp_interp_multi_channel_image = np.delete(temp_interp_multi_channel_image, 0, axis = 0)
#fig = plt.gcf()
## plt.imshow(temp_image)
## fig.set_size_inches(20,20)
#
#fig = plt.gcf()
#plt.imshow(temp_interp_multi_channel_image)
#plt.title("Bi-Linear interpolation image")
#fig.set_size_inches(20,20)
#
#cv2.imwrite(path_lab + '\\' + sample_number + "\\" + str(rect_W[0]) + ".tiff", cubic_image)
#