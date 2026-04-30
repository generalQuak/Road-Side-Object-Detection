Road-Side Object Detection / Classification Project
EECS 351 - Digital Signal Processing and Analysis

Project Website:
https://generalquak.github.io/Road-Side-Object-Detection/

Project Summary:
This project studies road object classification using MATLAB-based baseline classifiers and image feature extraction. We compare KNN, Naive Bayes, SVM, and Decision Tree models on labeled road-object images and analyze how dataset quality, class balance, and feature selection affect performance.

Main Demo File:
Run: run_demo.m

What run_demo.m does:
- Loads a small sample dataset included in this zip
- Preprocesses images
- Extracts selected features
- Trains and tests multiple classifiers
- Displays confusion matrices
- Displays an accuracy comparison chart

Other Important Files:
- baseline_classifiers.m
  Baseline classifier comparison using image features

- feature_classifier_experiments.m
  Classifier comparison using selectable feature extraction methods

- show_preprocessed_image.m
  Displays an example of the preprocessing pipeline on one image

Data Notes:
The full dataset used in development is too large to include in this zip.
A reduced sample dataset is included in:
dataset_sample/

Full dataset link:
https://drive.google.com/drive/folders/1X8OymxTTjpyPlPXYaQ2Venccy6lYp6zm?usp=sharing

How to run:
1. Open MATLAB in this project folder
2. Open run_demo.m
3. Make sure datasetPath points to "dataset_sample"
4. Run the script

Expected Output:
- Confusion matrices for each classifier
- Accuracy comparison bar chart
- Printed results table in the MATLAB command window