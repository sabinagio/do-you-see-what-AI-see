import os
import shutil
import random
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