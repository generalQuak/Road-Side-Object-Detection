%% feature_classifier_experiments.m
clear; clc; close all;

%% Settings
datasetPath = "dataset_sample";
imgSize = [64 64];              % slightly larger helps HOG/DFT
useGray = true;
runCNN = false;

% Choose ONE:
% "raw", "edge", "hog", "dft", "color", "edge_dft_color"
featureMode = "edge_dft_color";

cnnFinished = false;
accCNN = NaN;

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
    feat = extractSelectedFeatures(img, imgSize, useGray, featureMode);
    X(i,:) = feat(:)';
end

%% Split train/test
cv = cvpartition(Y, 'HoldOut', 0.2);

trainIdx = training(cv);
testIdx  = test(cv);

XTrain = X(trainIdx, :);
YTrain = Y(trainIdx, :);

XTest  = X(testIdx, :);
YTest  = Y(testIdx, :);

disp("Training set size:");
disp(size(XTrain,1));
disp("Test set size:");
disp(size(XTest,1));
disp("Feature dimension:");
disp(size(XTrain,2));

%% KNN
mdlKNN = fitcknn(XTrain, YTrain, ...
    'NumNeighbors', 3, ...
    'Standardize', 1);

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

%% CNN
if runCNN
    disp("Entering CNN block...");

    % Use the SAME train/test split as the classical models
    imdsTrain = subset(imds, find(trainIdx));
    imdsTest  = subset(imds, find(testIdx));

    inputSize = [64 64 3];
    augTrain = augmentedImageDatastore(inputSize, imdsTrain);
    augTest  = augmentedImageDatastore(inputSize, imdsTest);

    numClasses = numel(categories(imds.Labels));

    layers = [
        imageInputLayer(inputSize)

        convolution2dLayer(3,8,'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,16,'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,32,'Padding','same')
        batchNormalizationLayer
        reluLayer

        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', 8, ...
        'MiniBatchSize', 32, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', augTest, ...
        'Verbose', false, ...
        'Plots', 'training-progress');

    net = trainNetwork(augTrain, layers, options);

    predCNN = classify(net, augTest);
    YTestCNN = imdsTest.Labels;
    accCNN = mean(predCNN == YTestCNN);
    cnnFinished = true;

    figure;
    confusionchart(YTestCNN, predCNN);
    title(sprintf('CNN Accuracy = %.2f%%', accCNN * 100));

    disp("CNN accuracy:");
    disp(accCNN);
else
    disp("runCNN is false. Skipping CNN block.");
end

%% Compare results
modelNames = {'KNN'; 'Naive Bayes'; 'SVM'; 'Decision Tree'};
accuracies = [accKNN; accNB; accSVM; accTree];

if cnnFinished
    modelNames = {'KNN'; 'Naive Bayes'; 'SVM'; 'Decision Tree'; 'CNN'};
    accuracies = [accKNN; accNB; accSVM; accTree; accCNN];
end

resultsTable = table(modelNames, accuracies, ...
    'VariableNames', {'Model', 'Accuracy'});

disp("Accuracy summary:");
disp(resultsTable);

figure;
bar(accuracies * 100);
set(gca, 'XTickLabel', modelNames, 'XTickLabelRotation', 20);
ylabel('Accuracy (%)');
title(sprintf('Classifier comparison using %s features', featureMode));
grid on;

%% ---------------- Local functions ----------------

function feat = extractSelectedFeatures(img, imgSize, useGray, featureMode)

    % Preprocess
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
                feat = grayImg(:);
            else
                feat = rgbImg(:);
            end

        case "edge"
            feat = extractEdgeFeatures(grayImg);

        case "hog"
            feat = extractHOGFeatures(grayImg, 'CellSize', [8 8]);

        case "dft"
            feat = extractDFTFeatures(grayImg);

        case "color"
            feat = extractColorHistFeatures(rgbImg);

        case "edge_dft_color"
            f1 = extractEdgeFeatures(grayImg);
            f2 = extractDFTFeatures(grayImg);
            f3 = extractColorHistFeatures(rgbImg);
            feat = [f1(:); f2(:); f3(:)];

        otherwise
            error("Unknown featureMode.");
    end
end

function feat = extractEdgeFeatures(grayImg)
    % FIR-style spatial filters
    sobelX = [-1 0 1; -2 0 2; -1 0 1];
    sobelY = sobelX';

    gx = imfilter(grayImg, sobelX, 'replicate');
    gy = imfilter(grayImg, sobelY, 'replicate');

    mag = sqrt(gx.^2 + gy.^2);
    ang = atan2(gy, gx);

    % Simple edge statistics
    meanMag = mean(mag(:));
    stdMag  = std(mag(:));
    maxMag  = max(mag(:));

    % Orientation histogram
    numBins = 8;
    angleEdges = linspace(-pi, pi, numBins+1);
    histVals = histcounts(ang(:), angleEdges, 'Normalization', 'probability');

    % Edge density from binary detector
    bw = edge(grayImg, 'Canny');
    edgeDensity = mean(bw(:));

    feat = [meanMag; stdMag; maxMag; edgeDensity; histVals(:)];
end

function feat = extractDFTFeatures(grayImg)
    % 2D DFT magnitude features
    F = fftshift(fft2(grayImg));
    M = abs(F);
    M = M / (sum(M(:)) + eps);

    [rows, cols] = size(M);
    cx = floor(cols/2) + 1;
    cy = floor(rows/2) + 1;

    [X, Y] = meshgrid(1:cols, 1:rows);
    R = sqrt((X - cx).^2 + (Y - cy).^2);
    R = R / max(R(:));

    % Radial frequency bands
    bands = [0 0.15 0.3 0.5 0.75 1.0];
    bandEnergy = zeros(length(bands)-1,1);

    for k = 1:length(bands)-1
        mask = (R >= bands(k)) & (R < bands(k+1));
        bandEnergy(k) = sum(M(mask));
    end

    % Horizontal / vertical energy tendency
    centerStrip = 4;
    horizontalMask = abs(Y - cy) <= centerStrip;
    verticalMask   = abs(X - cx) <= centerStrip;

    horizEnergy = sum(M(horizontalMask));
    vertEnergy  = sum(M(verticalMask));

    feat = [bandEnergy; horizEnergy; vertEnergy];
end

function feat = extractColorHistFeatures(rgbImg)
    nbins = 8;

    r = rgbImg(:,:,1);
    g = rgbImg(:,:,2);
    b = rgbImg(:,:,3);

    hr = histcounts(r(:), nbins, 'Normalization', 'probability');
    hg = histcounts(g(:), nbins, 'Normalization', 'probability');
    hb = histcounts(b(:), nbins, 'Normalization', 'probability');

    feat = [hr(:); hg(:); hb(:)];
end