% Convert all PNG files in a folder to JPG and rename them (tag)##.jpg

clear; clc;

tag = ['pedestrian'];

inputFolder = tag;   % folder containing your .png files
outputFolder = [tag '_jpg'];  % folder to save renamed .jpg files
prefix = tag;

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

files = dir(fullfile(inputFolder, '*.png'));

for i = 1:length(files)
    % Read image
    imgPath = fullfile(inputFolder, files(i).name);
    img = imread(imgPath);

    % Handle PNG transparency if present
    if size(img,3) == 4
        img = img(:,:,1:3);
    end

    % Create new filename like pedestrian01.jpg
    newName = sprintf('%s%02d.jpg', prefix, i);
    outPath = fullfile(outputFolder, newName);

    % Write JPG
    imwrite(img, outPath, 'jpg');
end

disp('Conversion complete.');