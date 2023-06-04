This project aims to test a simple Convolutional Neural Net (CNN) architecture on the CIFAR10 dataset.

## Description
The CIFAR10 dataset is a widely-used dataset in the field of computer vision. It consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The goal of this project is to train a CNN model to classify these images into their respective classes.

We use a simple architecture and train the model on minimally pre-processed data without any data augmentation. The aim is to demonstrate that even with minimal modifications, a simple CNN can achieve reasonable accuracy on the CIFAR10 dataset. We achieve an accuracy of 75% on the validation set.

## Table of Contents 
TO DO

## Dependencies
* TensorFlow (tensorflow==2.11.0)
* TensorFlow Datasets (tensorflow-datasets==4.9.2)

## Usage
### Data Download
Run ```python3 tools/download_data.py``` to download the cifar10 dataset available through https://www.tensorflow.org/datasets/catalog/overview.

### Training the Model (Optional)
We have provided saved model weights inside the ```checkpoints``` folder. To retrain the model, run ```python3 train_model.py```.

### Inference
To generate a new prediction, run ```python3 predict.py "./checkpoints/training/cp-027.ckpt"```. You must provide a path to the checkpoint to load the pre-trained model weights. Your checkpoint path must end with **.ckpt**. Do not include the rest of the name (ex: .data-00000-of-00001). 
