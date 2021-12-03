#%%
import matplotlib.pyplot as plt
import os,sys,glob
import numpy as np
import PIL 
import envi
from PIL import Image
import shutil
# adding the parent directory to basic_utils
path = r"D:\Dropbox\Project\Project Flash\Code"
sys.path.append(path) 
from basic_utils.rm_utils import bandcsv_to_envi

#------------------------------ Define export_image function--------------------------------------------------
def export_image(envi_path, image_name, max_coeff = 0.15, colormap = 'gnuplot'):
    
    # reading envi file from the core
    # path = data_path + '\\' + cores_name + '\\' + envi_filename 
    
        # reading the aligned image
    envi_file = envi.envi(envi_path)
    image = envi_file.loadall()
    wavenumber = np.asarray(envi_file.header.wavelength)
    
    # if cores folder doesn't exist image folder, create one
    current_core_path = envi_path[:envi_path.find('Envi')]
    image_folder_path = current_core_path + 'image'
    if not os.path.exists(image_folder_path):
        os.mkdir(image_folder_path)
        
    # save as png file
    for i in range(len(wavenumber)):
        fname = image_folder_path + '\\'+ image_name + '_' + str(int(wavenumber[i])) +'.png'
        plt.imsave(fname, image[i], vmin = np.min(image[i]), vmax = max_coeff * np.max(image[i]), cmap = colormap ) # default :'gnuplot'
#------------------------------ Define csv2envi function--------------------
def csv2envi(path):
    working_path = path
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

        # after generate envi file, move all the csv file into raw folder
        raw_folder = path_data + '\\raw'
        if not os.path.exists(raw_folder):
            os.mkdir(raw_folder)
        for file in allcsvlist:
            shutil.move(file, raw_folder)
#------------------------------- Define mask function------------------------------
def mask(img, tissue_mask):
    # getting dimension of the image
    img_dimen = len(img.shape)
    # if it is one channel image
    if img_dimen == 2:
        channel = 1
        img_x = img.shape[1]
        img_y = img.shape[0]
    elif img_dimen == 3:
        channel = img.shape[0]
        img_x = img.shape[2]
        img_y = img.shape[1]
    else:
        raise ValueError("Error, the image has wrong dimensions")
    # check if the dimension match of the image and mask
    mask_x = tissue_mask.shape[1]
    mask_y = tissue_mask.shape[0]
    if not (img_x == mask_x and img_y == mask_y):
        raise ValueError("Error, the image and mask has unmatched dimensions")
    
    # put mask on the image
    mask = tissue_mask.astype(float)
    img = img.astype(float)
    if channel == 1:
        img_masked = img*mask
    else:
        img_masked = np.empty((channel, img_y, img_x))
        for i in range(channel):
            img_masked[i,:,:] = img[i]*mask
    return img_masked
#------------------------------ Define core fig save function------------------------------
def core_save(cores_number):
    
    save_path = r"D:\Dropbox\Project\Project Flash\Data\Core_Image"

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    # load in high-resolution image
    path = r"D:\Dropbox\Project\Project Flash\Data\high_resolution"
    coreslist = os.listdir(path)
    cores_name = coreslist[cores_number]
    cores_name_505 = cores_name + '_505'
    envi_high_resolution = envi.envi(path + '\\' + cores_name + '\\Envi' + cores_name)
    
    
    # load in aligned image
    path = r"D:\Dropbox\Project\Project Flash\Data\rect_pixels"
    envi_aligned = envi.envi(path + '\\' + cores_name_505 + '\\Envi' + cores_name + '_aligned')
    
    
    # load in sharpened image
    envi_sharpen = envi.envi(path + '\\' + cores_name_505 + '\\Envi' + cores_name + '_sharpen')
    
    
    # load in the mask
    path = r"D:\Dropbox\Project\Project Flash\Data\tissue_mask"

    tissue_mask = plt.imread(path + '\\' + cores_name + '_tm.png')
    tissue_mask = np.delete(tissue_mask, -1, 0)
    high_resolution_wavenumber = envi_high_resolution.header.wavelength
    band1660 = high_resolution_wavenumber.index(1660)
    alignment_wavenumber = envi_aligned.header.wavelength
    band1668 = alignment_wavenumber.index(1668)
    
    # load in the image
    image_high_resolution = envi_high_resolution.loadall()
    image_aligned = envi_aligned.loadall()
    image_sharpen = envi_sharpen.loadall()
    masked_image_high_resolution = mask(image_high_resolution, tissue_mask)
    masked_image_aligned = mask(image_aligned, tissue_mask)
    masked_image_sharpen = mask(image_sharpen, tissue_mask)

    # draw 3 images with same color scale min = 0, max = coef * max_image_high_resolution
    coef = 0.15
    cs_min = 0
    cs_max = coef * np.max(masked_image_high_resolution[band1660])

    # draw high-resolution image band 1660
    fig,ax = plt.subplots(3,1)
    ax[0].imshow(masked_image_high_resolution[band1660], vmin = cs_min, vmax = cs_max, cmap = 'gnuplot' )
    #ax[0].set_title(cores_name + '_high_resolution_band_1660')
    ax[0].axis('off')

    # draw algined image band 1668
    ax[1].imshow(masked_image_aligned[band1668], vmin = cs_min, vmax = cs_max, cmap = 'gnuplot' )
    #ax[1].set_title(cores_name + '_aligned_band_1668')
    ax[1].axis('off')

    # draw sharpened image band 1668
    ax[2].imshow(masked_image_sharpen[band1668], vmin = cs_min, vmax = cs_max, cmap = 'gnuplot' )
    #ax[2].set_title(cores_name + '_sharpen_band_1668')
    ax[2].axis('off')
    
    
    fig.patch.set_facecolor('black')
    fig.set_size_inches(30,90)

    fig.tight_layout()
    path = save_path + '\\' + cores_name + '.png'
    plt.savefig(path, dpi = 100)
    plt.clf()
    plt.cla()
    print(f"The {cores_number}, {cores_name} core image saved.")