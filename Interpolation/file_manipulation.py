#%% Deleting the wrong file############################################################
import os,glob,sys

path_data_lab = r"C:\Users\ReddyLabAdmin\Dropbox\Project\Project Flash\Data\csv_rect_pixels"
working_path = path_data_lab

cores = os.listdir(working_path)
for i in range(len(cores)):
    path_data = working_path + '\\' + cores[i] + '\\Envi' + cores[i][:cores[i].find('_')] + '0_5_pixel.HDR'
    if os.path.exists(path_data):
        os.remove(path_data)
    else:
        print("No such file exists")

# %% Moving all the .csv file into raw folder########################################
import os,glob,sys,shutil

rect_path_data_lab = r"C:\Users\ReddyLabAdmin\Dropbox\Project\Project Flash\Data\csv_rect_pixels"
orig_path_data_lab = r"C:\Users\ReddyLabAdmin\Dropbox\Project\Project Flash\Data\0.5_cerv_cores"
working_path = orig_path_data_lab


cores = os.listdir(working_path)
for i in range(len(cores)):
    path_data = working_path + '\\' + cores[i] + '\\raw'
    if not os.path.exists(path_data):
         os.mkdir(path_data)
    else:
         print("Folder already exists")
    cvslist = glob.glob(working_path + '\\' + cores[i] + '\\/*.csv')
    for file in cvslist:
        shutil.move(file, path_data)
# %% Remove all the wrong file and rename sharpened image########################################
import os,glob,sys

path = r"D:\Dropbox\Project\Project Flash\Data\csv_rect_pixels"

cores = os.listdir(path)
for i in range(len(cores)):
    sharpen_image_to_be_deleted = path + '\\' + cores[i] + '\\Envi' + cores[i][:cores[i].find('_')] + '_sharpen'
    header_file = sharpen_image_to_be_deleted + '.hdr' 
    if os.path.exists(sharpen_image_to_be_deleted):
        os.remove(sharpen_image_to_be_deleted)
    else:
        print("The file doesn't exists")
    if os.path.exists(header_file):
        os.remove(header_file)
    else:
        print("The header file doesn't exists")
    
    ### rename sharpened image
    sharpen_image_to_rename = path + '\\' + cores[i] + '\\Envi' + cores[i] + '_sharpen'
    sharpen_image_new_name = path + '\\' + cores[i] + '\\Envi' + cores[i][:cores[i].find('_')] + '_sharpen'
    header_file_to_rename = sharpen_image_to_rename + '.hdr'
    header_file_new_name = sharpen_image_new_name + '.hdr'
    if os.path.exists(sharpen_image_to_rename):
        os.rename(sharpen_image_to_rename, sharpen_image_new_name)
    if os.path.exists(header_file_to_rename):
        os.rename(header_file_to_rename, header_file_new_name)
# %% move all the envi_align and envi_sharpend into folder
import os,glob,sys
import shutil


file_path = r"D:\Dropbox\Project\Project Flash\Data\rect_pixels"
dest_path = r"D:\Dropbox\Project\Project Flash\Data\Aligned_and_sharpened"

core_list = [glob.glob(file_path + '\\' + core + '\\') for core in os.listdir(file_path)]
for number, each_core in enumerate(core_list):
   
    all_file = os.listdir(each_core[0])
    file_list = [ os.path.join(each_core[0], file) for file in all_file if ('_aligned' in file) or ('_sharpen' in file) ]
    
    for content in file_list:
        shutil.copy(content, dest_path)
    
   # if number == 0:
    #    break


# %%
