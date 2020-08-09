import warnings
warnings.filterwarnings('ignore')

import os
import shutil
import numpy as np
import tensorflow as tf

import argparse

from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument('input_path')
parser.add_argument('output_path')
args = parser.parse_args()

raw_image_dir= args.input_path
all_raw_files = np.array(os.listdir(raw_image_dir))

np.random.shuffle(all_raw_files)

def create_dir( path ):
    if not os.path.exists( path ):
        os.mkdir( path )
        print('Directory created:',path)
    else:
        print('Directory already exist:',path)

train_dir = './IMAGES_labelled/'

create_dir(train_dir)

def copy_files( source_dir, dest_dir, files_list ,file_ext='.jpg' ):
    for file in files_list:
        if file_ext in file:
            shutil.copy( os.path.join(source_dir,file), dest_dir)

classes = ['Pin_hole','Tensile_line','Water & Bright spot','Wrinkled', 'oil', 'thickness']

for img_class in classes:
    create_dir(os.path.join(train_dir,img_class))
    copy_files( raw_image_dir, os.path.join(train_dir,img_class), all_raw_files, img_class)


data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory(
        train_dir,
        target_size=(204, 204),
        batch_size=100,
        class_mode='categorical')

model = Sequential()
model.add(ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet'))
model.add(Dense(len(classes), activation = 'softmax'))
model.layers[0].trainable = False

sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])


cb_checkpointer  = ModelCheckpoint( filepath = os.path.join(args.output_path, 'model.hdf5'), mode = 'auto' )

fit_history = model.fit_generator(
        train_generator,
        steps_per_epoch=10,
        epochs = 3,
        callbacks=[cb_checkpointer]
)

shutil.rmtree(train_dir)
