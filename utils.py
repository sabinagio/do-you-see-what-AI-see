import os
import shutil
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create function to split data from glaucoma & normal folders
def split_data(source_dir, train_dir, val_dir, split_size):
    source_files = os.listdir(source_dir)

    # Ensure there are non-empty files
    files_to_copy = []

    for file_path in source_files:
        if os.path.getsize(os.path.join(source_dir, file_path)) > 0:
            files_to_copy.append(file_path)

    # Shuffle the files in the list for further random selection
    files_to_copy = random.sample(files_to_copy, len(files_to_copy))

    # Remove previous files from training & validation folders
    for file_path in os.listdir(train_dir):
        os.remove(os.path.join(train_dir, file_path))

    for file_path in os.listdir(val_dir):
        os.remove(os.path.join(val_dir, file_path))

    # Copy files to the training & validation set
    training_size = int(split_size * len(files_to_copy))
    for i in range(0, training_size):
        source_path = os.path.join(source_dir, files_to_copy[i])
        destination_path = os.path.join(train_dir, files_to_copy[i])
        shutil.copyfile(source_path, destination_path) 

    for i in range(training_size, len(files_to_copy)):
        source_path = os.path.join(source_dir, files_to_copy[i])
        destination_path = os.path.join(val_dir, files_to_copy[i])
        shutil.copyfile(source_path, destination_path) 

# Create image generators for the training & validation data
def image_generators(train_dir, val_dir, train_img_size, val_img_size):
  """
  Inputs:
  train_dir = training data directory
  val_dir = validation data directory
  train_img_size = the size of the training input images (tuple)
  val_img_size = the size of the validation input images (tuple)

  Outputs:
  train_generator = image generator for training data
  val_generator = image generator for validation data
  """

  # Instatiate ImageGenerator & rescale
  train_datagen = ImageDataGenerator(rescale=1./255)
  val_datagen = ImageDataGenerator(rescale=1./255)

  # Apply the ImageGenerator to the training & validation datasets
  train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                                      batch_size=20,
                                                      color_mode='rgb',
                                                      class_mode='binary',
                                                      target_size=train_img_size)

  val_generator = val_datagen.flow_from_directory(directory=val_dir,
                                                                batch_size=20,
                                                                color_mode='rgb',
                                                                class_mode='binary',
                                                                target_size=val_img_size)
  
  return train_generator, val_generator

# Define function to visualize results to determine if model is overfitting
def visualize_results(history):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  epochs = range(len(acc)) 
  ax, fig = plt.subplots(figsize=(20, 4))
  plt.subplot(121)
  plt.plot(epochs, acc, 'r', "Training Accuracy")
  plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
  plt.title('Training and validation accuracy')
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  plt.subplot(122)
  plt.plot(epochs, loss, 'r', "Training Loss")
  plt.plot(epochs, val_loss, 'b', "Validation Loss")
  plt.show()

# Read new images and save the true labels in a numpy array (0 - glaucoma, 1 - normal)
def get_testing_data(directories,  input_shape):
  """
  Args:
  directories = a list of directories
  input_shape = the desired shape of the images, expressed as (x, y), as the 
    color is inferred by the keras resize function
  
  Returns:
  X = a numpy array containing all of the resized images
  y = a numpy array containing all the labels based on the directory name
  """


  X = []
  y = []
  files = {}

  for directory in directories:
    files[directory] = os.listdir(directory)

  for directory in files.keys():
    for file in files[directory]:
      file_path = os.path.join(directory, file)
      image = tf.keras.preprocessing.image.load_img(file_path)
      image_data = tf.keras.preprocessing.image.img_to_array(image)
      image_data = tf.keras.preprocessing.image.smart_resize(image_data, input_shape)
      X.append(image_data)    

      if directory.split("/")[-1] == "glaucoma":
        y.append(0)
      elif directory.split("/")[-1] == "normal":
        y.append(1)

  X = np.array(X)
  y = np.array(y)

  return X, y

def load_acrima_model(model_name):
  model_dir = os.path.join("pre-trained-models", model_name)
  model_weights = os.path.join(model_dir, "_".join([model_name, "weights.h5"]))
  model_path = os.path.join(model_dir, "model.json")

  # Read and load model and weights
  json_file = open(model_path, 'r')
  model = json_file.read()
  json_file.close()

  model = tf.keras.models.model_from_json(model)
  model.load_weights(model_weights)
  model.summary()
  return model