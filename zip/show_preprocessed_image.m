%% show_preprocessed_image.m
% Shows one example image after the preprocessing steps used in the project

clear; clc; close all;

%% Settings
imagePath = "dataset_sample/vehicle/vehicle01.jpg";   % change this if needed
imgSize = [64 64];
saveFigure = true;

%% Read image
img = imread(imagePath);

% Resize original display for consistency
imgResized = imresize(img, imgSize);

% Convert to grayscale
if size(imgResized, 3) == 3
    grayImg = rgb2gray(imgResized);
else
    grayImg = imgResized;
end

grayImg = im2double(grayImg);

%% Edge features (Sobel + Canny)
sobelX = [-1 0 1; -2 0 2; -1 0 1];
sobelY = sobelX';

gx = imfilter(grayImg, sobelX, 'replicate');
gy = imfilter(grayImg, sobelY, 'replicate');
mag = sqrt(gx.^2 + gy.^2);

bwEdges = edge(grayImg, 'Canny');

%% DFT magnitude display
F = fftshift(fft2(grayImg));
M = log(1 + abs(F));   % log scale for visibility
M = mat2gray(M);

%% Show results
figure('Name', 'Preprocessing Example', 'NumberTitle', 'off');

subplot(2,3,1);
imshow(img);
title('Original Image');

subplot(2,3,2);
imshow(imgResized);
title('Resized Image');

subplot(2,3,3);
imshow(grayImg);
title('Grayscale');

subplot(2,3,4);
imshow(mat2gray(mag));
title('Sobel Magnitude');

subplot(2,3,5);
imshow(bwEdges);
title('Canny Edges');

subplot(2,3,6);
imshow(M);
title('DFT Magnitude');

sgtitle('Example of Project Preprocessing Pipeline');

%% Optional save
if saveFigure
    if ~exist("sample_outputs", "dir")
        mkdir("sample_outputs");
    end
    saveas(gcf, fullfile("sample_outputs", "preprocessed_example.png"));
    disp("Saved figure to sample_outputs/preprocessed_example.png");
end