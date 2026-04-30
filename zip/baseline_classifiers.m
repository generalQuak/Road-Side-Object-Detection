%% Baseline Image Classification Project
% Folder structure expected:
% dataset/
%   class1/
%   class2/
%   class3/
%
% Example:
% dataset/
%   pedestrian/
%   vehicle/
%   bicycle/
%   traffic_sign/

clear; clc; close all;

%% Load data
datasetPath = "dataset";   % <-- change this to dataset folder
imgSize = [32 32];         % resize all images to this size
useGray = false;            % set to false for RGB features 

imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

disp("Classes found:");
disp(categories(imds.Labels));
disp("Number of images:");
disp(countEachLabel(imds));

%% Preprocess
numImages = numel(imds.Files);

if useGray
    numFeatures = imgSize(1) * imgSize(2);
else
    numFeatures = imgSize(1) * imgSize(2) * 3;
end

X = zeros(numImages, numFeatures);
Y = imds.Labels;

for i = 1:numImages
    img = readimage(imds, i);

    % Convert to grayscale if requested
    if useGray
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
    else
        % If image is grayscale but RGB expected, replicate channels
        if size(img, 3) == 1
            img = cat(3, img, img, img);
        end
    end

    % Resize image
    img = imresize(img, imgSize);

    % Convert to double and flatten into row vector
    img = im2double(img);
    X(i, :) = img(:)';
end

% Split into train/test sets
cv = cvpartition(Y, 'HoldOut', 0.2);

XTrain = X(training(cv), :);
YTrain = Y(training(cv), :);

XTest  = X(test(cv), :);
YTest  = Y(test(cv), :);

disp("Training set size:");
disp(size(XTrain,1));
disp("Test set size:");
disp(size(XTest,1));

%% KNN
k = 3;

mdlKNN = fitcknn(XTrain, YTrain, ...
    'NumNeighbors', k, ...
    'Standardize', 1);

predKNN = predict(mdlKNN, XTest);
accKNN = mean(predKNN == YTest);

figure;
confusionchart(YTest, predKNN);
title(sprintf('KNN Confusion Matrix (Accuracy = %.2f%%)', accKNN*100));

%% Naive Bayes
mdlNB = fitcnb(XTrain, YTrain);

predNB = predict(mdlNB, XTest);
accNB = mean(predNB == YTest);

figure;
confusionchart(YTest, predNB);
title(sprintf('Naive Bayes Confusion Matrix (Accuracy = %.2f%%)', accNB*100));

%% SVM
% fitcsvm is naturally binary, so for multiclass we use ECOC with SVM learners
t = templateSVM('KernelFunction', 'linear', 'Standardize', true);

mdlSVM = fitcecoc(XTrain, YTrain, 'Learners', t);

predSVM = predict(mdlSVM, XTest);
accSVM = mean(predSVM == YTest);

figure;
confusionchart(YTest, predSVM);
title(sprintf('SVM (ECOC) Confusion Matrix (Accuracy = %.2f%%)', accSVM*100));

%% Tree
mdlTree = fitctree(XTrain, YTrain);

predTree = predict(mdlTree, XTest);
accTree = mean(predTree == YTest);

figure;
confusionchart(YTest, predTree);
title(sprintf('Decision Tree Confusion Matrix (Accuracy = %.2f%%)', accTree*100));

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
title('Baseline Classifier Comparison');
grid on;