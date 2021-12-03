#%%
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 20:15:27 2021

@author: Rupali
"""

import envi
import numpy 
import pandas 
import glob, os, sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
# adding the parent directory to basic_utils
path_home = r"D:\Dropbox\Project\Project Flash\Code"
path_lab = r"C:\Users\ReddyLabAdmin\Dropbox\Project\Project Flash\Code"
sys.path.append(path_lab)
  
from basic_utils.rm_utils import bandcsv_to_envi

#----------------------------convirting cvs files to Envi files----------------------------------------------
path_data = r"D:\Dropbox\Project\Project Flash\Data\rect_pixels"

working_path = path_data
cores = os.listdir(working_path)
for i in range(len(cores)):
    path_data = working_path +"\\" + cores[i]
    allcsvlist = glob.glob(path_data + "\\" + "/*.csv")
    csvlist = [f for f in allcsvlist if "OPTIR" in f]
    wavenumberlist = np.zeros(len(csvlist))
    fname = path_data +'\\Envi'+cores[i]
    for j in range(len(csvlist)):
        # wavenumberlist[j] = int(csvlist[j][-8:-4]) #if there is no number after band number
        wavenumberlist[j] = int(csvlist[j][-10:-6]) #if there is one number after band number
        BandsHSI = bandcsv_to_envi(csvlist,wavenumberlist,fname)

#%% converting 0.5x0.5 csv file into envi file
working_path = path_data_lab_0_5
cores = os.listdir(working_path)
for i in range(len(cores)):
    path_data = working_path +"\\" + cores[i]
    allcsvlist = glob.glob(path_data + "\\" + "/*.csv")
    csvlist = [f for f in allcsvlist if "OPTIR" in f]
    wavenumberlist = np.zeros(len(csvlist))
    fname = path_data +'\\Envi'+cores[i]
    for j in range(len(csvlist)):
        # wavenumberlist[j] = int(csvlist[j][-8:-4]) #if there is no number after band number
        wavenumberlist[j] = int(csvlist[j][-10:-6]) #if there is one number after band number
        BandsHSI = bandcsv_to_envi(csvlist,wavenumberlist,fname)

#%%
path1 = r"D:\Rect_pix_data\ovary\G6_305/"

# #read high resolution (square pixel envi file)
# sq = envi.envi(path1+'EnviG9')
# sqE = sq.loadall()
# sq_W = np.asarray(sq.header.wavelength)


#read high resolution (square pixel envi file)
rect = envi.envi(path1+'EnviG6_305')
rectE = rect.loadall()
rect_W = np.asarray(rect.header.wavelength)

# #find band images correspodnding to major amide bands 
# amide_1 = 1660
# a1_ind = np.argmin(np.abs(sq_W-amide_1))
# amide_2 = 1540
# a2_ind = np.argmin(np.abs(sq_W-amide_2))
# amide_3 = 1233
# a3_ind = np.argmin(np.abs(sq_W-amide_3))

# #select amide I band image from sq Envi file as a reference image
# Iref = sqE[a1_ind,:,:]  

# #resize images to Iref
# dim = (Iref.shape[1], Iref.shape[0])
# rect_inter = np.zeros([rectE.shape[0], dim[1], dim[0]], dtype=np.float32)
# for i in range(rectE.shape[0]):
#     rect_inter[i,:,:,] = cv2.resize(rectE[i,:,:],dim,interpolation = cv2.INTER_AREA)
    
# # align all the band images from rectE to amide I band image
# warp_affine = []

# for i in range(rectE.shape[0]):
#     rect_inter[i,:,:], warp_matrix = align_Images(rect_inter[i,:,:], Iref)
    
# rect = 'EnviG6_305'

# outfname  = path1+'EnviG9_305_inter'
# envi.save_envi(rect_inter, ''.join(outfname), 'BSQ',rect_W)

                