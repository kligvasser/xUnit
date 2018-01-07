scale = [3,4];
dataset = 'BSD/test/BSD100/';
apath = './images/';

imDir = fullfile(dataset,apath);
hrDir = fullfile(dataset,'gt_im');
lrDir = fullfile(dataset,'lr_im'); % ps

if ~exist(lrDir,'dir')
    mkdir(lrDir)
end

if ~exist(hrDir,'dir')
    mkdir(hrDir)
end

hrImgs = dir(fullfile(imDir,'*.*'));
for idx = 1:length(hrImgs)
    for sc = 1:length(scale)
        imgName = hrImgs(idx).name;
        try
            hrImg = imread(fullfile(imDir,imgName));
        catch
            disp(imgName);
            continue;
        end
        [h, w, c] = size(hrImg);
        ch = floor(h/scale(sc))*scale(sc);
        cw = floor(w/scale(sc))*scale(sc); %
        hrImg = hrImg(1:ch,1:cw,:);
        lrImg = imresize(hrImg,1/scale(sc),'bicubic');
        lrImg = imresize(lrImg,scale(sc),'bicubic'); % comment for ps
        [~,woExt,ext] = fileparts(imgName);
        lrName = sprintf('%sx%d%s',woExt,scale(sc),'.png');
        hrName = sprintf('%sx%d%s',woExt,scale(sc),'.png');
        imwrite(lrImg,fullfile(lrDir,lrName));
        imwrite(hrImg,fullfile(hrDir,hrName));
    end
    if mod(idx,100) == 0
        fprintf('Processed %d / %d images\n',idx,length(hrImgs));
    end
end
