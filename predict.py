import warnings
warnings.filterwarnings('ignore')

import os
import shutil
import numpy as np
import tensorflow as tf

import argparse
from PIL import Image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.resnet import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

import cv2
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('input_path')
parser.add_argument('model_path')
parser.add_argument('output_path')
args = parser.parse_args()

def normalize_shape(w):
    r = w/200
    r_mod = r%1
    if (r<=0.5):
        nw = 200
    elif (r_mod>0.5):
        nw = int(w+((1-(r%1))*200))
    else:
        nw = int(w-((r%1)*200))
    return(int(round(nw/10)*10))


def find_coordinates(x):
    c = []
    if x >= 200:
        ixc = 100
    else:
        ixc = int(x/2)
        c.append(ixc)
    for a in list(range(int(x/200))):
        c.append(ixc)
        ixc = ixc + 200
    if x-(c[-1]+100) > 0:
        c.append(int(x-100))
    return(c)

def patch_coordinates(w,c):
    wc = find_coordinates(w)
    cc = find_coordinates(c)
    coordinates = []
    for i in wc:
        for j in cc:
            coordinates.append(list((i,j)))

    w_step = len(wc)
    c_step = len(cc)

    return(coordinates, [w_step, c_step])

def make_patches(image):
    patches = []
    w = image.shape[0]
    c = image.shape[1]
    pc, steps = patch_coordinates(w,c)

    for i in pc:
        patch_center = np.array(i)
        patch_size = 200
        patch_x = int(patch_center[0] - patch_size / 2.)
        patch_y = int(patch_center[1] - patch_size / 2.)
        patch_image = image[patch_x:patch_x+patch_size,
                            patch_y:patch_y+patch_size]

        patches.append(patch_image)

    return(patches, pc, steps)

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def predict_image(image_folder):

    th = [0.53, 0.58, 0.89, 0.77, 0.92, 0.56]

    data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

    test_generator = data_generator.flow_from_directory(
    directory = image_folder,
    target_size = (224, 224),
    batch_size = 1,
    class_mode = None,
    shuffle = False,
    seed = 123)

    test_generator.reset()
    pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)

    image_label = []
    for i in pred:
        j = np.argmax(i)
        if i[j] >= th[j]:
            image_label.append(j)
        else:
            image_label.append(-1)

    return(image_label)


def make_border(patch, l):

    c_set = [[255,165,0],
            [255,192,203],
            [0, 128, 0],
            [183, 132, 167],
            [255,0,0],
            [255,255,0]]

    if l == -1:
        return(patch)

    else:

        bordersize = 4

        border = cv2.copyMakeBorder(
            patch[bordersize:-bordersize,bordersize:-bordersize],
            top=bordersize,
            bottom=bordersize,
            left=bordersize,
            right=bordersize,
            borderType=cv2.BORDER_CONSTANT,
            value=c_set[l]
        )

        return(border)

def combine(patches, steps, label_set):
    combine_image = {}
    c = 0
    for i in range(steps[0]):
        patch_bgr = cv2.cvtColor(patches[c], cv2.COLOR_GRAY2RGB)
        patch_bgr = make_border(patch_bgr, label_set[c])
        combine_image[i] = patch_bgr
        c=c+1
        for j in range(steps[1]-1):
            patch_bgr = cv2.cvtColor(patches[c], cv2.COLOR_GRAY2RGB)
            patch_bgr = make_border(patch_bgr, label_set[c])
            combine_image[i] = np.append(combine_image[i], patch_bgr, axis=1)
            c=c+1

    for i in list(combine_image.keys()):
        if i == 0:
            final_image = combine_image[0]
        else:
            final_image = np.append(final_image, combine_image[i], axis=0)

    return(final_image)

print('Initializing Model')
classes = ['Pin_hole','Tensile_line','Water & Bright spot','Wrinkled', 'oil', 'thickness']
model = Sequential()
model.add(ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet'))
model.add(Dense(len(classes), activation = 'softmax'))
#model.load_weights(args.model_path)

prediction_files = np.array(os.listdir(args.input_path))

create_dir(os.path.join(args.output_path, 'images_result'))

classes = ['Pin_hole','Tensile_line','Water & Bright spot','Wrinkled', 'oil', 'thickness']
res_df = pd.DataFrame(columns=classes)
fn_list = []

for filename in prediction_files:

    if '.jpg' in filename:
        print(filename)
        fn_list.append(filename)
        file_path = os.path.join(args.input_path, filename)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_resize = cv2.resize(image,(normalize_shape(image.shape[1]), normalize_shape(image.shape[0])))

        patches, centers, steps = make_patches(image_resize)


        create_dir(os.path.join(args.input_path, 'image_folder_main'))
        create_dir(os.path.join(args.input_path, 'image_folder_main', 'image_folder_sub'))


        for i, patch in enumerate(patches):
            cv2.imwrite(os.path.join(args.input_path, 'image_folder_main', 'image_folder_sub', str(i) + '_' + filename), patch)
        image_label = predict_image(os.path.join(args.input_path, 'image_folder_main'))
        shutil.rmtree(os.path.join(args.input_path, 'image_folder_main'))
        final_image = combine(patches, steps, image_label)
        image_original_size = cv2.resize(final_image,(image.shape[1], image.shape[0]))


        cv2.imwrite(os.path.join(args.output_path, 'images_result', 'result_'+filename), image_original_size)

        image_label = list(dict.fromkeys(image_label))


        label_onehot = []
        for i, c in enumerate(classes):
            if i in image_label:
                label_onehot.append(1)
            else:
                label_onehot.append(0)


        a_series = pd.Series(label_onehot, index = res_df.columns)
        res_df = res_df.append(a_series, ignore_index=True)



res_df.insert(0, 'file_name',fn_list)
res_df.to_csv(os.path.join(args.output_path, 'defects_results.csv'), index=False)

# for filename in prediction_files:
#
#     if '.jpg' in filename:
#         print(filename)
#         fn_list.append(filename)
#         file_path = os.path.join(input_path, filename)
#         image = cv2.imread(file_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         image_resize = cv2.resize(image,(normalize_shape(image.shape[1]), normalize_shape(image.shape[0])))
#
#         patches, centers, steps = make_patches(image_resize)
#
#
#         # create_dir(os.path.join(args.input_path, 'image_folder_main'))
#         # create_dir(os.path.join(args.input_path, 'image_folder_main', 'image_folder_sub'))
#         # create main and sub folder
#
#         for i, patch in enumerate(patches):
#             cv2.imwrite(os.path.join(sub, str(i) + '_' + filename), patch)
#
#         image_label = predict_image(main)
#
#         # shutil.rmtree(os.path.join(args.input_path, 'image_folder_main'))
#
#         final_image = combine(patches, steps, image_label)
#         image_original_size = cv2.resize(final_image,(image.shape[1], image.shape[0]))
#
#
#         cv2.imwrite(os.path.join(output_path, 'result_'+filename), image_original_size)
#
#         image_label = list(dict.fromkeys(image_label))
#
#
#         label_onehot = []
#         for i, c in enumerate(classes):
#             if i in image_label:
#                 label_onehot.append(1)
#             else:
#                 label_onehot.append(0)
#
#
#         a_series = pd.Series(label_onehot, index = res_df.columns)
#         res_df = res_df.append(a_series, ignore_index=True)
#
#
#
# res_df.insert(0, 'file_name',fn_list)
# res_df.to_csv(os.path.join(output_path, 'defects_results.csv'), index=False)
