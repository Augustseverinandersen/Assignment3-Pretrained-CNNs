
[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/Aj7Sf-j_)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10866021&assignment_repo_type=AssignmentRepo)

# Assignment 3 – Pretrained CNNs - Using pretrained CNNs for image classification
## Contribution
- The code in this assignment is inspired from the class notebooks, and from [Vijayabhaskar J](https://vijayabhaskar96.medium.com/tutorial-on-keras-flow-from-dataframe-1fd4493d237c). All comments have been written by me. The data used in this assignment is from [Rashmi Margani](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset), the Kaggle dataset author, refers to the datasets original authors [Pranjal Singh Rajput and Shivangi Aneja](https://arxiv.org/abs/2104.02830)
- This assignment uses the pretrained convolutional neural network VGG16. VGG16 has 16 conolutional layers and is used for transfer learning.

### Data 
- The dataset consists of 106 thousand images of 15 different  ethnic clothing items. The data is split up into three JSON files for train, test, and validation, and three folders containing the images, also split into train, test, and validation. The JSON files contain five headers *image_url, image_path, brand, product_title* and *class_label*. The makes uses of *image_path* and *class_label*. *Image_path* is the path to the image from the folder *images*. *Class_label* is the images label. The 15 possible labels are:

*Blouse, dhoti_pants, dupattas, gowns, kurta_men, leggings_and_salwars, lehenga, mojaris_men, mojaris_woman, nehru_jackets, palazzos, petticoats, saree, sherwanis,* and *woman_kurta.* 

- The folder train has 91166 images, folder val has 7500 images, and folder test has 7500 images. All images have different sizes. The images are gathered from Indian e-commerce websites. 

## Packages
- Matplotlib
  - Matplotlib is being used to create a visualization of the models training loss and accuracy curve
- Pandas
  - Pandas is being used to load the JSON files in as a pandas dataframe
- Scikit_learn
  - Scikit_learn is being used to create a classification report.
- Tensorflow
  - Tensorflow is being used to load VGG16 and create new layers. Furthermore, Tensorflow is being used to for the ImageDataGenerator including flow_from_dataframe. 
- Zipfile
  - Zipfile is being used to unpack the zipfile.
- Os
  - Is used to navigate filepaths.
- Argparse
  - Is used to create command-line arguments.
- Sys
  - Is used to navigate the directory.

## Repository contents
This repository consists of five folders and 3 files:
- assignment_3 (virtual environment)
- data (contains the data, when the zip file is extracted)
- figs (contains the plot created in this script)
- models (contain saved model, and classification report)
- src (contrains the script)
- README.md (Readme file)
- requirements.txt (required packages to install)
- bash setup.sh (Creates the virutal enviroment, installs requirements, and upgrades pip)

## Machine specifications and my usage 
- This script was created on Ucloud with a 32 CPU machine, and Coder Python 1.73.1. 
- This script took 1 hour and 40 minutes to run. Which consits of 7 minutes to un zip the zip file, and 90 minutes to train the model on 10 000 train images with 10 epochs.

## Assignment description
From Ross: Your instructions for this assignment are short and simple:
- You should write code which trains a classifier on this dataset using a pretrained CNN like VGG16
- Save the training and validation history plots
- Save the classification report

## Methods / what the code does
The script does the following: Unzips the zip file and loads the JSON files into pandas dataframe. Than an option of sampling the data is presented. Sample size can be specified with argsparse. The default sample size is all data. Next the ```ImageDataGenerator``` for the train and validation data is created with the following parameters: The images can be flipped horizontally, rotate 20 degress, but all images will be rescaled to 0-1. Than [Tensorflows](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_dataframe) ```flow_from_dataframe``` is created for the train and validation images. Images will be loaded in at pixel size 224 by 224 and a batch_size of 128. The train and validation images are than stored as a tuple of data containing two arrays of input features and labels. The size of the tuple depands on the batch size. Furthermore the labels from the JSON file will be one-hot encoded. Than the ```ImageDataGenerator``` for the test images is created, with the only parameter of rescaling the images. The script than uses ```flow_from_dataframe``` for the test images, where the output is stored the same way as for the train and validation images. The script than loads the pretrained model __VGG16__, and specifies an input shape of 224 by 244 with three colour channels. The model is loaded without classifier layers, and with the weights of the pre-trained layers unchangable. The new classifier layers consists of a flatten layer, a batch normalization layer, two layers with RelU activation and a softmax layer to make the output of the previous layer into a probability distribution over the 15 clothes items. The model is than trained with the training data simuntanisly with the validation data. The training of the model is than saved and is used to plot a loss and accuracy curve. The plot is created with matplotlib and saved in folder *figs*. Lastly, the model is tested with the training data and a classification report is created and saved to folder *models*.


## Findings
Sample
-	I worked on a subset of the data due to time limitations. I worked with 10 000 train images, 2000 validation images, and 2000 test images.

Plots 
-	The loss and accuracy plot is saved in the folder figs. The plot shows that the data is not over nor under fitting, but that train and validation are following each other smoothly. 

Classification report 
-	The classification report is saved in the folder models. The model has an f1 accuracy score of 0.69. The model preforms the worst on gowns (0.48), and best on blouse (0.87)

Model
-	The model is saved in the folder models

## Usage

To use this script follow these steps:
1.	Get the data from [Kaggle](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset) as a zip file and place it in the data folder. 
2.	Run ```bash setup.sh``` in the command line, which will create a virtual environment, and install the requirements.
3.	Run ```source ./assignment_3/bin/activate``` in the command-line, to activate the virtual environment.
4.	In the command line write this:
  - ```python3 src/testscript.py --zip_name data/archive.zip --filepath "data" --train_sample_size 1000 --val_sample_size 200 --test_sample_size 200 --epochs 5```
    -	```--zip_name``` = The path to your zip file including the zip file
    -	```--filepath``` = The path to the folder images excluding images. (located in data folder)
    -	The rest of the arguments all have a default value, but can be modified.
