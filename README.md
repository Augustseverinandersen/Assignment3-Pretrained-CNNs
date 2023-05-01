[here](https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator).

# Assignment 3 – Pretrained CNNs - Using pretrained CNNs for image classification
## Contribution
The code in this assignment is inspired from the class notebooks, and from [Vijayabhaskar J](https://vijayabhaskar96.medium.com/tutorial-on-keras-flow-from-dataframe-1fd4493d237c). All comments have been written by me. The data used in this assignment is gathered from [Rashmi Margani](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset), the Kaggle dataset author. 
The dataset consists of 106 thousand images in 15 different categories. The data is split up into three JSON files for train, test, and validation, and three folders containing the images, also split into train, test, and validation. The images are gathered from Indian e-commerce websites. 

This assignment used the pretrained convolutional neural network VGG16 *find link to VGG16*


## Assignment description
From Ross: Your instructions for this assignment are short and simple:
- You should write code which trains a classifier on this dataset using a pretrained CNN like VGG16
- Save the training and validation history plots
- Save the classification report

## Methods / what the code does
The code in this assignment does the following. Loads and unzips the data. Place the three JSON files into a pandas dataframe. Provides the possibility to create a sample size of the data. Creates an image data generator for the train and validation images, and one for the test images. Uses tensorflows flow_from_dataframe, to load the images and data augment them in real time. Loads a pretrained model, VGG16. Trains the model with the training data, and creates predictions with the test data. Lastly, it plots the model history and creates a classification report.

## Findings
Sample
-	I have chosen to work on a subset of the data due to time limitations. I worked with 10 000 train images, 2000 validation images, and 2000 test images.
Plots 
-	The loss and accuracy plot is saved in the folder figs. The plot shows that the data is not over nor under fitting, but that train and validation are following each other smoothly. 
Classification report 
-	The classification report is saved in the folder models. The model has an f1 accuracy score of 0.69. The model preforms the worst on gowns (0.48), and best on blouse (0.87)
Model
-	The model is saved in the folder models

## Usage

OBS! Ucloud was bugging. So, the first guide might not work, as I am getting the data from the zipfile. The second guide gets the data from the visual_analytics folder. But I am will work on this another day, when Ucloud is ready.

### Guide one
To use this script follow these steps:
1.	Get the data from Kaggle as a zip file and place it in a data folder 
2.	Run ```bash setup.sh``` in the command line, which will create a virtual environment, and install the necessary requirements.
3.	Run ```source ./assignment_3/bin/activate``` to activate the virtual environment.
4.	In the command line write this
  - python3 src/testscript.py --zip_name data/archive.zip --filepath "data" --train_sample_size 1000 --val_sample_size 200 --test_sample_size 200 --epochs 5
    -	```--zip_name``` = The path to your zip file including the zip file
    -	```--filepath``` = The path to the folder images excluding images. (located in data folder)
    -	The rest of the arguments all have a default value, but can be modified.

### Guide two
To use this script follow these steps:
1.	Run ```bash setup.sh``` in the command line, which will create a virtual environment, and install the necessary requirements.
2.	Run ```source ./assignment_3/bin/activate``` to activate the virtual environment.
3.	In the command line write this
  -	python3 src/testscript.py --zip_name data/archive.zip --filepath "../../images" --train_sample_size 1000 --val_sample_size 200 --test_sample_size 200 --epochs 5
