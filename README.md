
[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/Aj7Sf-j_)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10866021&assignment_repo_type=AssignmentRepo)

# 3. Assignment 3 – Pretrained CNNs - Using pretrained CNNs for image classification
## 3.1 Assignment Description
Written by Ross: 
Your instructions for this assignment are short and simple, you should write code that trains a classifier on this dataset using a pre-trained CNN like VGG16. Save the training and validation history plots and the classification report.
## 3.2 Machine Specifications and My Usage
All the computation done for this project was performed on the UCloud interactive HPC system, which is managed by the eScience Center at the University of Southern Denmark. Python 1.73.1 with 32 CPU. This script took 1 hour and 40 minutes to run. 7 minutes to unpack the zip file, and 90 minutes to train the model on 10 000 train images with 10 epochs.
### 3.2.1 Prerequisites 
To run this script, make sure to have Bash and Python 3 installed on your device. This script has only been tested on Ucloud. 
## 3.3 Contribution
The code in this assignment is inspired by [Vijayabhaskar J.](https://vijayabhaskar96.medium.com/tutorial-on-keras-flow-from-dataframe-1fd4493d237c) All comments have been written by me. The data used in this assignment is from Kaggle user [Rashmi Margani](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset), but made by authors [Pranjal Singh Rajput and Shivangi Aneja](https://arxiv.org/abs/2104.02830).
This assignment uses the pre-trained convolutional neural network VGG16. VGG16 has 16 convolutional layers and is used for transfer learning.
### 3.3.1 Data
The dataset consists of 106 thousand images of 15 ethnic clothing items (_Blouse, dhoti pants, dupattas, gowns, kurta men, leggings and salwars, lehenga, mojaris men, mojaris woman, nehru jackets, palazzos, petticoats, saree, sherwanis, and woman kurta_). The zip file contains three JSON files (_train, test, and validation_), a folder called images, which contains three folders (_train, test, and validation_), which contain the images. The JSON files have five headers _image_url, image_path, brand, product_title , and class_label_. The script uses _image_path_ and _class_label_. The column _image_path_ contains the path to the image from the folder images. The column _class_label_ contains the image's label. The folder train has 91166 images, folder val has 7500 images, and folder test has 7500 images. All images are of different sizes and gathered from Indian e-commerce websites.
## 3.4 Packages
These are the packages used in this script:
-	Matplotlib (version 3.7.1) is being used to create a visualization of the model's training loss and accuracy curve.
-	Pandas (version 2.0.1) is being used to load the JSON files in as a Pandas data frame and to sample the data.
-	Scikit-learn (version 1.2.2) is being used to create a _classification report_.
-	TensorFlow (version 2.12.0) is being used to load _VGG16_ and create new layers. Furthermore, TensorFlow is being used to import the _ImageDataGenerator_, which loads and augments the images, and _flow_from_dataframe_ functions.
-	Zipfile is being used to unpack the zip file.
-	Os is used to navigate file paths, on different operating systems.
-	Argparse is used to create command line arguments.
-	Sys is used to navigate the directory.
## 3.5 Repository contents
This repository contains the following folders and files:
-	**data** this is the empty folder, where the zip file will be placed.
-	**figs** this folder contains the plots created in this script.
-	**models** this folder contains the saved model and the classification report.
-	**src** this folder contains the script.
-	**README.md** the README file.
-	**requirements.txt** the text file with the required packages to install.
-	**setup.sh** this file creates the virtual environment, installs the requirements, and upgrades pip.
## 3.6 Methods 
The script does the following: 
- Unzips the zip file and loads the JSON files into Pandas data frames. 
- You are then given the option to create a sample size with argparse. The default is all the data. 
- Next, the _ImageDataGenerator_ for the training and validation data is created with the following parameters: 
    - The images can be flipped horizontally, 
    - Rotated 20 degrees.
    - Will be rescaled to 0-1. 
- Then [Tensorflows](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_dataframe) _flow_from_dataframe_ is created for the training and validation images. Images will be loaded in as pixel size 224 by 224 and a batch size of 128. The labels will be one-hot encoded.
    - The train and validation images are then stored as a tuple of data containing two arrays of input features and labels. The size of the tuple depends on the batch size. 
- Then the _ImageDataGenerator_ for the test images is created, with the only parameter of rescaling the images. 
- The script then uses _flow_from_dataframe_ for the test images, where the output is stored the same way as for the train and validation images. 
- The script then loads the pre-trained model _VGG16_ and specifies an input shape of 224 by 244 with three colour channels. The model is loaded without classifier layers and with the weights of the pre-trained layers unchangeable. The new classifier layers consist of a flatten layer, a batch normalization layer, two layers with relu activation, and a softmax layer to make the output of the previous layer into a probability distribution of the 15 clothes items. 
- The model is then trained with the training data simultaneously with the validation data to see the model’s performance during training. 
- The model’s training history is saved and used to plot a _loss and accuracy curve_. The plot is created with matplotlib and saved in the folder _figs_. 
- Lastly, the model is tested with the training data and a classification report is created and saved to folder _models_.
## 3.7 Discussion
Due to computation and time limitations, I worked on a sample size of the data. I decided to work on 10000 train images (out of 91166), 2000 validation images (out of 7500), 2000 test images (out of 7500), and a total of 10 epochs. The smaller sample size may have impacted the accuracy of my model. The reason for that is, that a larger dataset allows the model to learn diverse features from a wider range of images. Having fewer images gives the model a smaller chance to learn diverse features of images from the same class. 

The validation size of 20% of the training size, is an appropriate size for seeing how the model is learning. However, more test images would have given a more precise classification report, to see the model’s performance. Based on my classification report, each class had approximately 140 images to make predictions on. The _accuracy f1-score_ was 0.69 which shows that the model was accurate on 69% of the test images. The authors of the dataset had an accuracy of [88.43%](https://arxiv.org/abs/2104.02830). Giving the model more training images could have led to a higher score. The model performs the worst on class gowns (0.48), and best on class blouse (0.87).

Looking at the loss and accuracy plots for the model’s training history shows that the train and validation follow each other nicely. It shows that the model is not overfitting, but that it could have been trained on a few more epochs since the training curve had not flattened out. 
## 3.8 Usage 
To use this script, follow these steps:
1.	Clone the repository.
2.	Get the data from [Kaggle](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset) as a zip file and place it in the _data_ folder.
3.	Run ``bash setup.sh`` in the command line. This will create a virtual environment and install the requirements.
4.	Run ``source ./assignment_3/bin/activate`` in the command-line, to activate the virtual environment.
5.	In the command line write this ``python3 src/testscript.py --zip_name data/archive.zip --filepath "data" --train_sample_size 1000 --val_sample_size 200 --test_sample_size 200 --epochs 5``
    - The argparse ``--zip_name`` takes a string as input. Here you must write the path to your zip file.
    - The argparse ``--filepath`` takes a string as input and has the default data. Here you must write the path to the folder images excluding images. Only change this if the folder images is located somewhere else.
    - The argparse ``--train_sample_size`` takes an integer as input and has the default 91116. Change this if you want to reduce the size of the dataset.
    - The argparse ``--val_sample_size`` takes an integer as input and has the default 7500. Change this if you want to reduce the size of the dataset.
    - The argparse ``--test_sample_size`` takes an integer as input and has the default 7500. Change this if you want to reduce the size of the dataset.
    - The argparse ``--epochs`` takes an integer as input and has the default 10. Change this if you want to reduce or increase the training time.

