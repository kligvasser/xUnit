function bicubic_subsample()
%% matlab code to genetate mod images, bicubic-downsampled LR, bicubic_upsampled images.

%% set parameters
% comment the unnecessary line
input_folder = '';
save_mod_folder = '';
save_LR_folder = '';
% save_bic_folder = '';

up_scale = 3;
mod_scale = up_scale;

if exist('save_mod_folder', 'var')
    if exist(save_mod_folder, 'dir')
        disp(['It will cover ', save_mod_folder]);
    else
        mkdir(save_mod_folder);
    end
end
if exist('save_LR_folder', 'var')
    if exist(save_LR_folder, 'dir')
        disp(['It will cover ', save_LR_folder]);
    else
        mkdir(save_LR_folder);
    end
end
if exist('save_bic_folder', 'var')
    if exist(save_bic_folder, 'dir')
        disp(['It will cover ', save_bic_folder]);
    else
        mkdir(save_bic_folder);
    end
end

idx = 0;
filepaths = dir(fullfile(input_folder,'*.*'));
for i = 1 : length(filepaths)
    [paths,imname,ext] = fileparts(filepaths(i).name);
    if isempty(imname)
        disp('Ignore . folder.');
    elseif strcmp(imname, '.')
        disp('Ignore .. folder.');
    else
        idx = idx + 1;
        str_rlt = sprintf('%d\t%s.\n', idx, imname);
        fprintf(str_rlt);
        % read image
        img = imread(fullfile(input_folder, [imname, ext]));
        img = im2double(img);
        % modcrop
        img = modcrop(img, mod_scale);
        if exist('save_mod_folder', 'var')
            imwrite(img, fullfile(save_mod_folder, [imname, '.png']));
        end
        % LR
        im_LR = imresize(img, 1/up_scale, 'bicubic');
        if exist('save_LR_folder', 'var')
            imwrite(im_LR, fullfile(save_LR_folder, [imname, '.png']));
        end
        % Bicubic
        if exist('save_bic_folder', 'var')
            im_B = imresize(im_LR, up_scale, 'bicubic');
            imwrite(im_B, fullfile(save_bic_folder, [imname, '_bicx4.png']));
        end
    end
end
end

%% modcrop
function img = modcrop(img, modulo)
if size(img,3) == 1
    sz = size(img);
    sz = sz - mod(sz, modulo);
    if mod(modulo, 1) == 0  
        img = img(1:sz(1), 1:sz(2));
    end
else
    tmpsz = size(img);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    if mod(modulo, 1) == 0 
        img = img(1:sz(1), 1:sz(2),:);
    end
end
    
end
