#%%
import matplotlib.pyplot as plt
import os,sys,glob
import numpy as np
import PIL 
import envi
from PIL import Image
#%%
# Plot aligned image

path_data_home = r"D:\Dropbox\Project\Project Flash\Data\csv_rect_pixels"
path_data_lab = r"C:\Users\ReddyLabAdmin\Dropbox\Project\Project Flash\Data\csv_rect_pixels"

working_path = path_data_lab# setting data path
cores = os.listdir(working_path) # read all the cores from the folder

bandnumber = 8
cores_name = cores[4] # select the first core A10

# reading envi file from the core
path = working_path + '\\' + cores_name + '\\Envi' + cores_name[:cores_name.find('_')] + '_aligned'

    # reading the aligned image
envi_aligned = envi.envi(path)
aligned_image = envi_aligned.loadall()
aligned_wavenumber = np.asarray(envi_aligned.header.wavelength)

# save as png file
fname = path + '_'+ str(int(aligned_wavenumber[bandnumber])) +'.png'
plt.imsave(fname, aligned_image[bandnumber], vmin = np.min(aligned_image[bandnumber]), vmax = 0.15*np.max(aligned_image[bandnumber]), cmap = 'gnuplot')
# %% Plot original image
path_orig_data_path = r"D:\Dropbox\Project\Project Flash\Data\0.5_cerv_cores"
path_orig_data_path_lab = r"C:\Users\ReddyLabAdmin\Dropbox\Project\Project Flash\Data\0.5_cerv_cores"
working_path = path_orig_data_path_lab
cores = os.listdir(working_path)

bandnumber = 6 
cores_name = cores[4] # select the same core A10

# reading envi file from the core
path = working_path + '\\' + cores_name + '\\' + 'Envi'+ cores_name 

    # reading the aligned image
envi_original = envi.envi(path)
original_image = envi_original.loadall()
original_wavenumber = np.asarray(envi_original.header.wavelength)# gnuplot2 gnuplot coolwarm hotred

# save as png file
fname = path + '_ori_'+ str(int(original_wavenumber[bandnumber])) + '.png' 
plt.imsave(fname, original_image[bandnumber], vmin = np.min(original_image[bandnumber]), vmax = 0.1* np.max(original_image[bandnumber]), cmap = 'gnuplot')
# %% Plot only interpolated image
path_data_home = r"D:\Dropbox\Project\Project Flash\Data\csv_rect_pixels"
path_data_lab = r"C:\Users\ReddyLabAdmin\Dropbox\Project\Project Flash\Data\csv_rect_pixels"

working_path = path_data_lab# setting data pathworking_path = path_orig_data_path_lab
cores = os.listdir(working_path) # read all the cores from the foldercores = os.listdir(working_path)

bandnumber = 8 
cores_name = cores[4] # select the same core A10

# reading envi file from the core
path = working_path + '\\' + cores_name + '\\' + 'Envi'+ cores_name[:cores_name.find('_')] + '0_5_pixel'

    # reading the aligned image
envi_just_interp = envi.envi(path)
interp_image = envi_just_interp.loadall()
interp_wavenumber = np.asarray(envi_just_interp.header.wavelength)# gnuplot2 gnuplot coolwarm hotred

# save as png file
fname = path + '_' + str(int(interp_wavenumber[bandnumber])) + '.png' 
plt.imsave(fname, interp_image[bandnumber], vmin = np.min(interp_image[bandnumber]), vmax = 0.1* np.max(interp_image[bandnumber]), cmap = 'gnuplot')
# %% Plot rect image
path_data = r"D:\Dropbox\Project\Project Flash\Data\csv_rect_pixels"

working_path = path_data# setting data pathworking_path = path_orig_data_path_lab
cores = os.listdir(working_path) # read all the cores from the foldercores = os.listdir(working_path)

bandnumber = 8 
cores_name = cores[4] # select the same core A10

# reading envi file from the core
path = working_path + '\\' + cores_name + '\\' + 'Envi'+ cores_name

    # reading the aligned image
envi_rect = envi.envi(path)
rect_image = envi_rect.loadall()
rect_wavenumber = np.asarray(envi_rect.header.wavelength)# gnuplot2 gnuplot coolwarm hotred

# save as png file
fname = path + '_rect_' + str(int(interp_wavenumber[bandnumber])) + '.png' 
plt.imshow(fname, rect_image[bandnumber], vmin = np.min(rect_image[bandnumber]), vmax = 0.1* np.max(rect_image[bandnumber]), cmap = 'gnuplot')
#%%
sys.path.append(r"D:\Dropbox\Project\Project Flash\Code")
from basic_utils.xw_utils import export_image
#%%
path_data = r"D:\Dropbox\Project\Project Flash\Data\Aligned_and_sharpened"
#cores = os.listdir(path_data)

#core_number = 2 
fname = 'B9_sharpen' 
file_path = path_data + '\\Envi'  + fname 
export_image(file_path, fname)
#%%
path_data = r"D:\Dropbox\Project\Project Flash\Data\Masked"
fname = 'B9_sharpen_masked'
file_path = path_data + '\\Envi' + fname
export_image(file_path, fname, 0.1)
# %%
