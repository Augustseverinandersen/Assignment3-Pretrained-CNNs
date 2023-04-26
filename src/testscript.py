# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
# cifar10 data - 32x32
from tensorflow.keras.datasets import cifar10

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
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# for plotting
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import sys
sys.path.append(".")



def input_parse():
    # initialize the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str) # argument is filepath as a string
    parser.add_argument("--train_sample_size", type = int, default = 91166)
    parser.add_argument("--val_sample_size", type = int, default = 7500)
    parser.add_argument("--test_sample_size", type = int, default = 7500)
    parser.add_argument("--epochs", type = int, default = 10)
    args = parser.parse_args()

    return args





def loading_metadata(args):
    print("Loading metadata to dataframes")
    test_metadata = pd.read_json(os.path.join( args.filepath, "metadata", "test_data.json"), lines=True) # Loading in JSON files of metadata
    train_metadata = pd.read_json(os.path.join(args.filepath, "metadata", "train_data.json"), lines=True)
    val_metadata = pd.read_json(os.path.join(args.filepath, "metadata", "val_data.json"), lines=True)
    return test_metadata, train_metadata, val_metadata


def sampling(test_metadata, train_metadata, val_metadata):
    test_metadata = test_metadata.sample(1000)
    train_metadata = train_metadata.sample(2000)
    val_metadata = val_metadata.sample(1000)
    return test_metadata, train_metadata, val_metadata



def image_data_generator():
    # Data augmentaion 
    # ImageDataGenerator from tensorflow 
    datagen = ImageDataGenerator(horizontal_flip=True, # Flip it horizontally around the access
                                rotation_range=20, # Rotate the image randomly 20 degress around the access
                                rescale = 1/255 # rescale it between 0-1
    )
    # Take your images, create a pipelie (Take an image modify it, pass it on)
    return datagen



def image_directory(args):
    main_dir = os.path.dirname(args.filepath)
    directory_images = main_dir # removing the last part of the specified filepath
    return directory_images




def training_images(train_metadata, directory_images):
    train_tf = datagen.flow_from_dataframe( # using keras flow  from dataframe 
        dataframe = train_metadata, # Defining dataframe 
        directory = directory_images, # Path to images 
        x_col = "image_path", # rest of the image path from dataframe 
        y_col = "class_label", # column label
        subset = "training", # what this data is 
        target_size=(224, 224), # image should be loaded in as size 
        color_mode="rgb", # colors 
        class_mode = "categorical", # One hot encoding the labels 
        batch_size = 128, # take images of batchs 128 at a time
        shuffle = True # shuffle the images around 
    )
    return train_tf


def val_images(val_metadata, directory_images):
    val_tf = datagen.flow_from_dataframe(
        dataframe = val_metadata,
        directory = directory_images,
        x_col = "image_path",
        y_col = "class_label",
        #subset = "validation",
        target_size=(224, 224),
        color_mode="rgb",
        class_mode = "categorical",
        batch_size = 128,
        shuffle = True
    )
    return val_tf





def test_image_generator():
    test_datagen = ImageDataGenerator(
                                    rescale = 1./255. # datagenerator for test, it only has to rescale the images 
    )
    return test_datagen





def test_images(test_metadata, directory_images):
    test_tf = test_datagen.flow_from_dataframe(
        dataframe = test_metadata,
        directory = directory_images,
        x_col = "image_path",
        target_size=(224, 224),
        color_mode="rgb",
        class_mode = None,
        batch_size = 128,
        shuffle = False # do not shuffle the images 
    )





def load_model():  
    # load model without classifier layers
    model = VGG16(include_top=False, 
                pooling='avg',
                input_shape=(224, 224, 3))

    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
        
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    bn = BatchNormalization()(flat1) # Added batnormalization from tensorflow. Take the previouslayer, normalise the values, and than pass them on
    class1 = Dense(256, 
                activation='relu')(bn) # Added new classification layer 
    class2 = Dense(128, 
                activation='relu')(class1)
    output = Dense(15, # 15 labels
                activation='softmax')(class2)

    # define new model
    model = Model(inputs=model.inputs, 
                outputs=output)

    # compile
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)

    model.compile(optimizer=sgd,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    # summarize
    print(model.summary())
    return model




def train_model(train_tf, val_tf, args):
    print("Training model")
    # fit the data generator to our images

    # fits the model on batches with real-time data augmentation:

    H = model.fit( # fitting the model to 
        train_tf, # training tensorflow dataframe 
        steps_per_epoch = len(train_tf), # take as many steps as the length of the dataframe 
        validation_data = val_tf, # Validation data 
        validation_steps = len(val_tf), 
        epochs = args.epochs
    )
    # Possible to get image (stream the image) (stream it frow the dataGenerator) from the folder, instead of loading the image into the script. 
    return H



def plot_history(H, epochs, save_path):
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(save_path, "loss_and_accuracy_curve.png"))



def plot_model_history(H, args, save_path):
    print("Saving plots")
    save_path = os.path.join("..", "figs")
    plot_history(H, 10, save_path)







def prediction(test_tf, train_tf):
    print("Test model with test data")
    # Predict the label of the test_images
    pred = model.predict(test_tf)
    pred = np.argmax(pred,axis=1)

    # Map the label 
    labels = (train_tf.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    pred1 = [labels[k] for k in pred]
    return pred, pred1
    # Display the result
    #print(f'The first 5 predictions: {pred[:20]}')





def classification_report_function(test_metadata, pred):
    print("Classification Report: ")
    y_true = test_metadata['class_label'].values
    classification_report_test = classification_report(y_true, pred)
    print("Classification Report - Test Data:")
    print(classification_report_test)



def main_function():
  args = input_parse()
  test_metadata, train_metadata, val_metadata = loading_metadata(args)
  test_metadata, train_metadata, val_metadata = sampling(test_metadata, train_metadata, val_metadata)
  datagen = image_data_generator()
  directory_images = image_directory()
  train_tf = training_images(train_metadata, directory_images)
  val_tf = val_images(val_metadata, directory_images)
  test_datagen = test_image_generator()
  test_tf = test_images(test_metadata, directory_images)
  model = load_model()
  H = train_model(train_tf, val_tf, args)
  plot_history(H, epochs)
  plot_model_history(H, args, save_path)
  pred, pred1 = predictions(test_tf, train_tf)
  classification_report_function(test_metadata, pred)


# filepath "..", "..", "..", "images"




if __name__ == "__main__":
    main_function()









