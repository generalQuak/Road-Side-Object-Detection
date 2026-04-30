%% run_demo.m
% Simple demonstration for EECS 351 final project
clear; clc; close all;

%% Settings
datasetPath = "dataset_sample";   % small included dataset
imgSize = [64 64];
useGray = true;

% Choose: "raw", "edge", "dft", "edge_dft_color"
featureMode = "edge_dft_color";

%% Load data
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

disp("Classes found:");
disp(categories(imds.Labels));
disp("Number of images:");
disp(countEachLabel(imds));

Y = imds.Labels;
numImages = numel(imds.Files);

%% Build feature matrix
sampleImg = readimage(imds, 1);
sampleFeat = extractSelectedFeatures(sampleImg, imgSize, useGray, featureMode);
numFeatures = numel(sampleFeat);

X = zeros(numImages, numFeatures);

for i = 1:numImages
    img = readimage(imds, i);
    X(i,:) = extractSelectedFeatures(img, imgSize, useGray, featureMode);
end

%% Train/test split
cv = cvpartition(Y, 'HoldOut', 0.2);

XTrain = X(training(cv), :);
YTrain = Y(training(cv), :);
XTest  = X(test(cv), :);
YTest  = Y(test(cv), :);

disp("Training set size:");
disp(size(XTrain,1));
disp("Test set size:");
disp(size(XTest,1));
disp("Feature dimension:");
disp(size(XTrain,2));

%% KNN
mdlKNN = fitcknn(XTrain, YTrain, 'NumNeighbors', 3, 'Standardize', 1);
predKNN = predict(mdlKNN, XTest);
accKNN = mean(predKNN == YTest);

figure;
confusionchart(YTest, predKNN);
title(sprintf('KNN (%s) Accuracy = %.2f%%', featureMode, accKNN*100));

%% Naive Bayes
mdlNB = fitcnb(XTrain, YTrain);
predNB = predict(mdlNB, XTest);
accNB = mean(predNB == YTest);

figure;
confusionchart(YTest, predNB);
title(sprintf('Naive Bayes (%s) Accuracy = %.2f%%', featureMode, accNB*100));

%% SVM
t = templateSVM('KernelFunction', 'linear', 'Standardize', true);
mdlSVM = fitcecoc(XTrain, YTrain, 'Learners', t);
predSVM = predict(mdlSVM, XTest);
accSVM = mean(predSVM == YTest);

figure;
confusionchart(YTest, predSVM);
title(sprintf('SVM (%s) Accuracy = %.2f%%', featureMode, accSVM*100));

%% Decision Tree
mdlTree = fitctree(XTrain, YTrain);
predTree = predict(mdlTree, XTest);
accTree = mean(predTree == YTest);

figure;
confusionchart(YTest, predTree);
title(sprintf('Decision Tree (%s) Accuracy = %.2f%%', featureMode, accTree*100));

%% Compare results
modelNames = {'KNN'; 'Naive Bayes'; 'SVM'; 'Decision Tree'};
accuracies = [accKNN; accNB; accSVM; accTree];

resultsTable = table(modelNames, accuracies, ...
    'VariableNames', {'Model', 'Accuracy'});

disp("Model comparison:");
disp(resultsTable);

figure;
bar(accuracies * 100);
set(gca, 'XTickLabel', modelNames, 'XTickLabelRotation', 20);
ylabel('Accuracy (%)');
title(sprintf('Classifier Comparison using %s features', featureMode));
grid on;

%% ---------------- Local functions ----------------
function feat = extractSelectedFeatures(img, imgSize, useGray, featureMode)

    if size(img,3) == 3
        rgbImg = im2double(imresize(img, imgSize));
        grayImg = rgb2gray(rgbImg);
    else
        grayImg = im2double(imresize(img, imgSize));
        rgbImg = cat(3, grayImg, grayImg, grayImg);
    end

    switch string(featureMode)
        case "raw"
            if useGray
                feat = grayImg(:)';
            else
                feat = rgbImg(:)';
            end

        case "edge"
            feat = extractEdgeFeatures(grayImg);

        case "dft"
            feat = extractDFTFeatures(grayImg);

        case "edge_dft_color"
            f1 = extractEdgeFeatures(grayImg);
            f2 = extractDFTFeatures(grayImg);
            f3 = extractColorHistFeatures(rgbImg);
            feat = [f1(:); f2(:); f3(:)]';

        otherwise
            error("Unknown featureMode.");
    end
end

function feat = extractEdgeFeatures(grayImg)
    sobelX = [-1 0 1; -2 0 2; -1 0 1];
    sobelY = sobelX';

    gx = imfilter(grayImg, sobelX, 'replicate');
    gy = imfilter(grayImg, sobelY, 'replicate');

    mag = sqrt(gx.^2 + gy.^2);
    ang = atan2(gy, gx);

    meanMag = mean(mag(:));
    stdMag  = std(mag(:));
    edgeDensity = mean(edge(grayImg, 'Canny'), 'all');

    numBins = 8;
    binEdges = linspace(-pi, pi, numBins+1);
    histVals = histcounts(ang(:), binEdges, 'Normalization', 'probability');

    feat = [meanMag; stdMag; edgeDensity; histVals(:)];
end

function feat = extractDFTFeatures(grayImg)
    F = fftshift(fft2(grayImg));
    M = abs(F);
    M = M / (sum(M(:)) + eps);

    [rows, cols] = size(M);
    cx = floor(cols/2) + 1;
    cy = floor(rows/2) + 1;

    [X, Y] = meshgrid(1:cols, 1:rows);
    R = sqrt((X - cx).^2 + (Y - cy).^2);
    R = R / max(R(:));

    bands = [0 0.15 0.3 0.5 0.75 1.0];
    bandEnergy = zeros(length(bands)-1,1);

    for k = 1:length(bands)-1
        mask = (R >= bands(k)) & (R < bands(k+1));
        bandEnergy(k) = sum(M(mask));
    end

    feat = bandEnergy;
end

function feat = extractColorHistFeatures(rgbImg)
    nbins = 8;
    r = histcounts(rgbImg(:,:,1), nbins, 'Normalization', 'probability');
    g = histcounts(rgbImg(:,:,2), nbins, 'Normalization', 'probability');
    b = histcounts(rgbImg(:,:,3), nbins, 'Normalization', 'probability');
    feat = [r(:); g(:); b(:)];
end