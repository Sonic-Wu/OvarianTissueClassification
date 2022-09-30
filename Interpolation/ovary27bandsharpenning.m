%%
clear all;
data_path_save_image = "D:\Dropbox\Project\Project Flash\Data\ovary_27_wavenumbers_csv\Data\Sharpened";
%#list all the interpolated image cores############################################################
data_path_sharpen_image = "D:\Dropbox\Project\Project Flash\Data\ovary_27_wavenumbers_csv\Data\Alignment";
cores = dir(data_path_sharpen_image);
cores(1:2) = [];

%#list all the high-resolution image cores##########################################################
data_path_home_high_reso_image = "D:\Dropbox\Project\Project Flash\Data\ovary_27_wavenumbers_csv\Data\0_5X0_5";
cores_high_reso = dir(data_path_home_high_reso_image);
cores_high_reso(1:2) = [];
%runcell = {'I1_505', 'I2_505','I5_505'};

%#for each pair of cores, load envi files of aligned and high-resolution image######################
for i = 1:size(cores)
    %if ~strcmp(cores(i).name,'G8_505')
     %   continue;
    %end
    %if ~any(strcmp(runcell, cores(i).name))
    %    continue;
    %end    
%#####setting the same core for interpolated image and high-resolutionimage#########################
    tic
    %core_interp_full_name = cores(i).name;
    %core_interp_core_name = erase(core_interp_full_name, '_505');
    core_interp_core_name = cores(i).name;
    
    for j = 1:size(cores_high_reso)
        if strcmp(core_interp_core_name, cores_high_reso(j).name)
            core_high_reso_core_name = cores_high_reso(j).name;
            break;  
        end
        if j == size(cores_high_reso)
            fprintf('No corresponding core for %f', core_interp_core_name);
            error('No matching core for %f', core_interp_core_name);
        end
    end
    
    
%#####read envi for interpolated image##############################################################
    
    % set filename for itnerpolation and high-resolution image
    fname_envi_interp = data_path_sharpen_image + '\' + core_interp_core_name + '\Envi'+ core_interp_core_name + '_aligned_05_fft_inter_winguassian';
    fname_envi_high_reso = data_path_home_high_reso_image + ['\' ...
        ''] + core_high_reso_core_name + '\Envi'+ core_high_reso_core_name;
    
    % read envi file
    [envi_interp,envi_interp_header] = xwuenviLoadRaw(fname_envi_interp);
    [envi_high_reso, envi_high_reso_header] = xwuenviLoadRaw(fname_envi_high_reso);
    
    % doing the sharpenning via curvelets for each band
    fprintf('start to sharpen the %d core, %s .\n', i, core_interp_core_name);
    envi_image_sharpened = rmcurvelets(envi_interp, envi_high_reso(:,:,1));
    envi_image_sharpened = single(envi_image_sharpened);

    
    % save sharpened image into envi file
    fname_envi_sharpen = data_path_save_image + '\' + core_interp_core_name + '\Envi' + core_interp_core_name + '_sharpen_05_fft_inter_winguassian_fusionmethod2';
    headername_envi_sharpen = fname_envi_sharpen + '.hdr';
    xwenviSaveRaw(envi_image_sharpened, fname_envi_sharpen,headername_envi_sharpen, envi_interp_header.wavelength)
    toc
end


%% Regenerate header file
clear all;

%#list all the sharpen image cores############################################################
data_path_sharpen_image = "D:\Dropbox\Project\Project Flash\Data\csv_rect_pixels";
cores = dir(data_path_sharpen_image);
cores(1:2) = [];

%#for each cores, load envi files of sharpen image######################
for i = 1:size(cores)
    
    
%#####selected core and transfer the core name##########################
    core_full_name = cores(i).name;
    core_name = erase(core_full_name, '_505');   
    
%#####read envi of sharpen image##############################################################
    
    % set filename for itnerpolation and sharpen image
    fname_envi_interp = data_path_sharpen_image + '\' + core_full_name + '\Envi'+ core_name + '_aligned';
    fname_envi_sharpen = data_path_sharpen_image + '\' + core_full_name + '\Envi'+ core_name + '_sharpen';
    
    % read envi file
    %[envi_interp,envi_interp_header] = enviLoadRaw(fname_envi_interp);
    [envi_sharpen, envi_sharpen_header] = enviLoadRaw(fname_envi_sharpen, fname_envi_interp + '.hdr');
    
    % save the envi file of sharpen image with correct header file
    fname_envi_sharpen = data_path_sharpen_image + '\' + core_full_name + '\Envi' + core_name + '_sharpen';
    headername_envi_sharpen = fname_envi_sharpen + '.hdr';
    xwenviSaveRaw(envi_sharpen, fname_envi_sharpen, headername_envi_sharpen, envi_sharpen_header.wavelength);
    if i == 0
        break
    end
    
end
%%
tic
A = rand(12000,4400);
B = rand(12000,4400);
toc