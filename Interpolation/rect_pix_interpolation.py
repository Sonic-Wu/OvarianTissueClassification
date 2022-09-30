#%%
"""
This code is used for doing zero padding interpolation on hyperspectral Mirage images with 27 bands ovary
"""

from cv2 import CAP_PROP_OPENNI_MAX_TIME_DURATION
import envi
import numpy as np
import glob, os, sys
from scipy.signal import windows
import matplotlib.pyplot as plt 
import shutil
import cv2
import time


# adding Rupali's util into the directory
path = r"D:\Dropbox\Project\Project Flash\Code"
sys.path.append(path) 

from basic_utils.xw_utils import reshape_to_ref, inter_interleave, generate_1Dwindow, inter_fft_window, csv2envi
from basic_utils.rm_utils import align_Images


#%% 
# ------------------------------converting csv file to Envi file------------------------------ 
endo_csv_path_4x0_5 = r'D:\Dropbox\Project\Project Flash\Data\endo_27_wavenumbers\4X05'
endo_csv_path_5x0_5 = r"D:\Dropbox\Project\Project Flash\Data\endo_27_wavenumbers\5X05"
csv2envi(endo_csv_path_4x0_5)
csv2envi(endo_csv_path_5x0_5)
#%%
# ------------------------------loading the path------------------------------
rect_path = r'D:\Dropbox\Project\Project Flash\Data\endo_27_wavenumbers\4X05'
high_resolution_path = r"D:\Dropbox\Project\Project Flash\Data\endo_27_wavenumbers\05X05"

rect_pixel_list= os.listdir(rect_path)
high_resolution_pixel_list = os.listdir(high_resolution_path)

rect_pixel_list.sort()
high_resolution_pixel_list.sort()

runlist = []
#------------------------------run interpolation------------------------------
for sample_number in range(len(rect_pixel_list)):
    
    # get core name
    core_name = rect_pixel_list[sample_number]
    
    print(f"starting the {sample_number + 1}th core aligning... (total {len(rect_pixel_list)})")
    print(f"The current core name is: {core_name}")
    if core_name in runlist:
        print("Skip")
        continue
    # set file name
    try:
        index = high_resolution_pixel_list.index(core_name)
    except ValueError:
        print(f"{core_name}" + " doesn't have corresponding reference image")
        continue
    
    #ref_sample_name = high_resolution_pixel_list[index]
    if core_name in high_resolution_pixel_list:
        rect_envi_path = rect_path + '//' + core_name + '//Envi' + core_name
        ref_envi_path = high_resolution_path + '//' + core_name + '//Envi' + core_name
    else:
        print(f"{core_name}" + " doesn't have corresponding reference image")
        continue
    
    # read rect image envi file and ref image envi file
    envi_rect = envi.envi(rect_envi_path)
    envi_ref = envi.envi(ref_envi_path)

    # reference high_resolution image
    ref_hyperspectral_image = envi_ref.loadall()
    ref_hyperspectral_image = ref_hyperspectral_image.reshape((ref_hyperspectral_image.shape[1],ref_hyperspectral_image.shape[2]))
    ref_wavenumber = np.asarray(envi_ref.header.wavelength)
    ref_x, ref_y = envi_ref.header.samples, envi_ref.header.lines

    # rectagular image
    rect_hyperspectral_image = envi_rect.loadall()
    rect_wavenumber = np.asarray(envi_rect.header.wavelength)
    rect_x, rect_y = envi_rect.header.samples, envi_rect.header.lines
    pix_size = (0.5, 5)


    #------------------------------resizing rect pix image using interleaved zero padding------------------------------ 
    ## Iref_new, rect_inter, rect_Scale_band = inter_interleave(Iref, rectE, pixsize, keep_high_res_shape)

    #------------------------------interpolation using zero padding of fft------------------------------
    
    wintype = 'guassian'
    option = 150
    Rect_inter = inter_fft_window(rect_hyperspectral_image, pix_size, wintype, option)
    keep_high_res_shape = True
    warp_affine = []
    amide_ind = [np.argmin(np.abs(rect_wavenumber - 1660))] # there is only 1660 band image in the reference image
    print("Start aligning....\n")
    if keep_high_res_shape:
        print("Getting warp_matrix processing...\n")
        start_time = time.time()
        sz = ref_hyperspectral_image.shape # Find size of reference image
        rect_Scale_band = Rect_inter[amide_ind[0],:,:]
        rect_alignment = np.zeros((len(rect_wavenumber),sz[0],sz[1]))
        im_new_size = cv2.resize(rect_Scale_band,sz,interpolation = cv2.INTER_AREA)
        aligned_img, warp_matrix = align_Images(im_new_size, ref_hyperspectral_image)
        
        end_time = time.time()
        runtime = int(end_time - start_time)
        print(f"The warp_matrix takes {runtime} seconds." + '\n')
        
        #align rest of the bands
        for i in range(len(rect_wavenumber)):
            print("Start alignment...\n")
            print(f"start the {i+1}th channel {rect_wavenumber[i]} band aligning...")
            start_time = time.time()
            #resizing FTIR image 
            im_new_size = cv2.resize(Rect_inter[i,:,:],sz,interpolation = cv2.INTER_AREA)
     
            im_aligned = cv2.warpAffine(im_new_size, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
            rect_alignment[i,:,:] = im_aligned
            
            end_time = time.time()
            runtime = int(end_time - start_time)
            print(f"The {i + 1}th channel {rect_wavenumber[i]} band takes {runtime} seconds to align." + '\n')
       # rect_intern_ref = np.concatenate((rect_intern, np.reshape(Iref,(1,Iref.shape[0],Iref.shape[1]))), axis=0)
    else:
        
        Iref_new = cv2.resize(ref_hyperspectral_image, (Rect_inter.shape[2], Rect_inter.shape[1]), interpolation = cv2.INTER_AREA)
        rect_Scale_band = Rect_inter[amide_ind[0],:,:]
        Iref_new, warp_matrix = align_Images(Iref_new, rect_Scale_band)  # reference is interpolated image
        
        rect_intern_ref = np.concatenate((Rect_inter, np.reshape(Iref_new,(1,Iref_new.shape[0],Iref_new.shape[1]))), axis=0)
   
    #outfname  = rect_path +'/'+wintype+str(option)+'/'+'Envi_'+str(p)+'05_fft_inter_win'+wintype
    rect_envi_save_path =  r'D:\Dropbox\Project\Project Flash\Data\endo_27_wavenumbers\Alignment'
    if not os.path.exists(rect_envi_save_path):
        os.mkdir(rect_envi_save_path)
    rect_envi_save_path += '//' + core_name
    if not os.path.exists(rect_envi_save_path):
        os.mkdir(rect_envi_save_path)
    file_name_alignment = rect_envi_save_path + '//Envi' + core_name + '_aligned_' + '05_fft_inter_win' + wintype
    envi.save_envi(rect_alignment.astype(np.float32), file_name_alignment, wavelength = rect_wavenumber)

print("FINISH")
# %%
