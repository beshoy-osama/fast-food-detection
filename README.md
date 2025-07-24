# fast-food-detection
Fast Food Classification
This project builds a deep learning image classifier to identify different kinds of fast food items using TensorFlow and transfer learning with ResNet50.

Project Overview
The model classifies fast food images into 10 classes:

Baked Potato

Burger

Crispy Chicken

Donut

Fries

Hot Dog

Pizza

Sandwich

Taco

Taquito

A custom dataset is used with separate folders for Train, Validation, and Test sets.

Dataset
The dataset is organized into subfolders corresponding to each food class in the following directories:

text
/Train
/Valid
/Test
Each directory contains image files for the respective class.

How It Works
Images are loaded from dataset folders and labeled according to folder names.

Images undergo data augmentation (rotations, flips, zooms, shifts) during training to improve robustness.

The dataset is split into training (65%) and validation (35%) sets.

Uses transfer learning with a pre-trained ResNet50 model (ImageNet weights).

A custom classification head with dropout and batch normalization layers is added for fine-tuning.

The model is trained to classify images into one of 10 food categories.

Dependencies
Python 3.x

TensorFlow 2.x

pandas

scikit-learn

matplotlib (optional, for visualization)

Training Results
Training accuracy: 90.26%

Validation accuracy: 86.11%

Files
FastFoodApp.py — Main application script

FastFoodDetection.keras — Saved Keras model (Notexist)

FastFoodDetection_ResNet50.ipynb — Jupyter notebook with training code and analysis