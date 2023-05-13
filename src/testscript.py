# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (VGG16)
# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

#scikit-learn
from sklearn.metrics import classification_report

# Data munging
import matplotlib.pyplot as plt
import pandas as pd
import zipfile

# System tools
import os
import argparse
import sys
sys.path.append(".")


def input_parse():
    # Command line arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_name", type=str, help = "Name of the zip folder")
    parser.add_argument("--filepath", type=str, help = "Path to folder images, not including metadata, train, val, and test") # argument is filepath as a string
    parser.add_argument("--train_sample_size", type = int, default = 91166, help = "Sample size of training images, default is all images")
    parser.add_argument("--val_sample_size", type = int, default = 7500, help = "Sample size of validation images, default is all images")
    parser.add_argument("--test_sample_size", type = int, default = 7500, help = "Sample size of test images, default is all images")
    parser.add_argument("--epochs", type = int, default = 10, help = "How many epochs, default is 10")
    args = parser.parse_args()

    return args


def unzip(args):
    folder_path = os.path.join("..", "data", "images") # Path to the data if unzipped already
    if not os.path.exists(folder_path): # Checking to see if folder is unzipped
        print("Unzipping file")
        path_to_zip = args.zip_name # Defining the path to the zip file
        zip_destination = os.path.join("..", "data") # defining the output destination

        with zipfile.ZipFile(path_to_zip,"r") as zip_ref: # using the package from zipfile, to un zip the zip file
            zip_ref.extractall(zip_destination)
    print("The files are unzipped")


def loading_metadata(args):
    print("Loading metadata to dataframes") # Loading JSON files of metadata, containing labels and path to images
    test_metadata = pd.read_json(os.path.join(args.filepath, "test_data.json"), lines=True) 
    train_metadata = pd.read_json(os.path.join(args.filepath, "train_data.json"), lines=True)
    val_metadata = pd.read_json(os.path.join(args.filepath, "val_data.json"), lines=True)
    return test_metadata, train_metadata, val_metadata


def sampling(test_metadata, train_metadata, val_metadata, args):
    print("sampling") # Sample rows of dataframe, to make the data smaller.
    test_metadata = test_metadata.sample(args.test_sample_size)
    train_metadata = train_metadata.sample(args.train_sample_size)
    val_metadata = val_metadata.sample(args.val_sample_size)
    return test_metadata, train_metadata, val_metadata



def image_data_generator(): # Data augmentaion 
    print("Creating Image data generator")
    # ImageDataGenerator from tensorflow 
    datagen = ImageDataGenerator(horizontal_flip=True, # Flip it horizontally around the access randomly 
                                rotation_range=20, # Rotate the image randomly 20 degress around the access
                                rescale = 1/255 # rescale the pixel values to between 0-1
    )
    return datagen



def image_directory(args):  # removing the last part of the specified filepath
    print("Finding path to images") # the metadata contains the rest of the path hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh
    directory_images = os.path.join(args.filepath)
    return directory_images



def training_images(datagen, train_metadata, directory_images):
    print("Finding training images: ")
    train_tf = datagen.flow_from_dataframe( # using keras flow  from dataframe 
        dataframe = train_metadata, # Defining dataframe 
        directory = directory_images, # Path to images. The rest of the path is contained in the train_metadata
        x_col = "image_path", # Rest of the image path from dataframe 
        y_col = "class_label", # Column label
        subset = "training", # What this data is be used for. 
        target_size=(224, 224), # Specifying what size the images should be loaded in as 
        color_mode="rgb", # Images are red green blue. 
        class_mode = "categorical", # One hot encoding the labels 
        batch_size = 128, # Specifying batch size
        shuffle = True # Shuffle the images around 
    )
    return train_tf # Returning a tensorflow dataframe


def val_images(datagen, val_metadata, directory_images):
    print("Finding validation images: ")
    val_tf = datagen.flow_from_dataframe(
        dataframe = val_metadata, # Defining dataframe 
        directory = directory_images, # The first part of the path to the images 
        x_col = "image_path", # Image path
        y_col = "class_label", # Labels
        target_size=(224, 224), # Image size
        color_mode="rgb", # Color channels 
        class_mode = "categorical", # One hot encoding the labels 
        batch_size = 128, # Specifying batch size
        shuffle = True # Shuffle the images
    )
    return val_tf



def test_image_generator(): # Creating a datagenerator for the test images.
    print("Creating test image data generator")
    test_datagen = ImageDataGenerator(
                                    rescale = 1./255. # Datagenerator for test, it only has to rescale the images 
    )
    return test_datagen





def test_images(test_datagen, test_metadata, directory_images):
    print("Finding test images")
    test_tf = test_datagen.flow_from_dataframe(
        dataframe = test_metadata, # Test dataframe 
        directory = directory_images, # Path to the images 
        x_col = "image_path", # Path to the images
        target_size=(224, 224), # Image size
        color_mode="rgb", # Color images 
        class_mode = None, # No labels to be onehot encoded
        batch_size = 128, # Batch size
        shuffle = False # Do not shuffle the images 
    )
    return test_tf




def load_model(): # Code taken from in class notebooks
    print("Loading model: ")  
    # load model without classifier layers
    model = VGG16(include_top=False, # Exclude classifier layers
                pooling='avg',
                input_shape=(224, 224, 3)) # Input shape of the images. 224 pixels by 224. 3 color channels

    # Keep pretrained layers, and don't modify them
    for layer in model.layers:
        layer.trainable = False
        
    # Add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    bn = BatchNormalization()(flat1) # Added batchnormalization from tensorflow. Take the previouslayer, normalise the values, and than pass them on
    class1 = Dense(256, 
                activation='relu')(bn) # Added new classification layer 
    class2 = Dense(128, 
                activation='relu')(class1) 
    output = Dense(15, # 15 labels # Added new classification layer with 15 outputs. 15 labels in total
                activation='softmax')(class2)

    # define new model
    model = Model(inputs=model.inputs, 
                outputs=output)

    # compile
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01, # Start learning rate at 0.01
        decay_steps=10000, # Every 10 000 steps start decaying 
        decay_rate=0.9) # DEcay by 0.9 to the start learning rate
    sgd = SGD(learning_rate=lr_schedule)

    model.compile(optimizer=sgd,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    # summarize
    print(model.summary())
    return model




def train_model(train_tf, val_tf, args, model):
    print("Training model")

    H = model.fit( # fitting the model to 
        train_tf, # training data from tensorflow dataframe 
        steps_per_epoch = len(train_tf), # Take as many steps as the length of the dataframe 
        validation_data = val_tf, # Validation data from tensorflow dataframe
        validation_steps = len(val_tf), # Validation steps as length of validation data 
        epochs = args.epochs
    )
    return H # Return the training history



def plot_history(H, epochs, save_path): # Code taken from Ross from inclass notebook, but modified a bit
    plt.style.use("seaborn-colorblind")
    # First plot 
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    # Second plot
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(save_path, "loss_and_accuracy_curve.png")) # Save the plot as *name*



def plot_model_history(H, args): # Saving the plot
    print("Saving plots")
    save_path = os.path.join("figs") # Save in folder figs
    plot_history(H, args.epochs, save_path) # Plot function with model history, amount of epochs, and savepath


def predictions(test_tf, train_tf, model): # Code taken from source in readme file
    # Testing the model
    print("Testing model with test images")
    # Predict the label of the test_images
    pred = model.predict(test_tf) # Using test data on the model
    pred = np.argmax(pred,axis=1)

    # Map the label 
    labels = (train_tf.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    pred = [labels[k] for k in pred]
    return pred


def classification_report_function(test_metadata, pred): # Creating a classification report
    y_true = test_metadata['class_label'].values # Test labels from dataframe
    classification_report_test = classification_report(y_true, pred) # Create classification report with labels, and predictions
    print("Classification Report - Test Data:")
    print(classification_report_test)
    return classification_report_test


def saving_model(model): # Saving model
    print("Saving model")
    folder_path = os.path.join("models", "image_classification.keras") # Defining out path
    tf.keras.models.save_model( # Using Tensor Flows function for saving models.
    model, folder_path, overwrite=True, save_format=None 
    ) # Model name, folder, Overwrite existing saves, save format = none 


def metrics_save_function(classifier_metrics): # Saving classification report
    print("Saving classifier metrics")
    folder_path = os.path.join("models") # Saving in folder models
    file_name = "classifier_metrics.txt" # Name of the file
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, "w") as f: # "Writing" the classifier metrics, thereby saving it.
        f.write(classifier_metrics)
    print("Reports saved")


def main_function():
  args = input_parse() # Arguments
  unzip(args)
  test_metadata, train_metadata, val_metadata = loading_metadata(args) # Loading dataframes
  test_metadata, train_metadata, val_metadata = sampling(test_metadata, train_metadata, val_metadata, args) # Sampling the dataframes
  datagen = image_data_generator() # Datagenerator for train and validation
  directory_images = image_directory(args) # First part of the path to the images
  train_tf = training_images(datagen, train_metadata, directory_images) # Training data flow from dataframe 
  val_tf = val_images(datagen, val_metadata, directory_images) # Validation data flow from dataframe
  test_datagen = test_image_generator() # Test image datagenerator 
  test_tf = test_images(test_datagen, test_metadata, directory_images) # Test data flow from dataframe
  model = load_model() # Defining pretrained model
  H = train_model(train_tf, val_tf, args, model) # Training model
  plot_model_history(H, args) # Plotting
  pred = predictions(test_tf, train_tf, model) # Predicting on test data
  classification_report_test = classification_report_function(test_metadata, pred) # Classification report
  saving_model(model) # Saving model
  metrics_save_function(classification_report_test) # Saving classification report



if __name__ == "__main__": # If called from commandline run main function
    main_function()









