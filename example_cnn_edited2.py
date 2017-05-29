'''
example cnn from kaggle, edited somewhat by moi
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import math
from glob import glob
import os
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split

# from multiprocessing import Pool, cpu_count
from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
np.random.seed(1337)  # for reproducibility

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

def load_data_labels(filepath):
    '''
    Input: filepath for file containing labels (training data classifiers).
    Output: numpy array of inputted file (in this case, col1 = 'name' (int), col2 = 'invasive' (1/0 boolean))
    '''
    labels = pd.read_csv(filepath)
    return labels

def load_images(root, labels, test=False):
    '''
    Input: root where images are located, labels (training data classifiers).
    Output: List of path names (project) for images; numpy array of labels in same order as images in list of path names.
    '''
    file_paths_list = []
    test_names = []
    y = []
    for i in range(len(labels)):
        file_paths_list.append(root + str(int(labels.iloc[i][0])) + '.jpg')
        y.append(labels.iloc[i][1])
        if test:
            test_names.append(int(labels.iloc[i][0]))
    y = np.array(y)
    return file_paths_list, y, test_names

def _center_image(img, new_size=[256, 256]):
    '''
    Helper function. Takes rectangular image resized to be max length on at least one side and centers it in a black square.
    Input: Image (usually rectangular - if square, this function is not needed).
    Output: Image, centered in square of given size with black empty space (if rectangular).
    '''
    row_buffer = (new_size[0] - img.shape[0]) // 2
    col_buffer = (new_size[1] - img.shape[1]) // 2
    centered = np.zeros(new_size + [img.shape[2]], dtype=np.uint8)
    centered[row_buffer:(row_buffer + img.shape[0]), col_buffer:(col_buffer + img.shape[1])] = img
    return centered

def color_shift_image(img, color_method=cv2.COLOR_BGR2RGB):
    '''
    Transforms image color in method specified (accepts cv2 methods).
    '''
    return cv2.cvtColor(img, color_method)

def resize_image(img, new_size=[256, 256]):
    '''
    Resizes images without changing aspect ratio. Centers image in square black box.
    Input: Image, desired new size (new_size = [height, width]))
    Output: Resized image, centered in black box with dimensions new_size
    '''
    if(img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1]*256/img.shape[0]),256)
    else:
        tile_size = (256, int(img.shape[0]*256/img.shape[1]))
    return _center_image(cv2.resize(img, dsize=tile_size), new_size)

def crop_image(img, crop_size=[224, 224]):
    '''
    Crops image to new_dims, centering image in frame.
    Input: Image, desired cropped size (crop_size=[height, width])
    Output: Cropped image
    '''
    row_buffer = (img.shape[0] - crop_size[0]) // 2
    col_buffer = (img.shape[1] - crop_size[1]) // 2
    return img[row_buffer:(img.shape[0] - row_buffer), col_buffer:(img.shape[1] - col_buffer)]

def process_images(file_paths_list):
    '''
    Input: list of file paths (images)
    Output: numpy array of processed images (color shifted, resized, centered)
    '''
    x = []
    for file_path in file_paths_list:
        img = cv2.imread(file_path)
        img = color_shift_image(img)
        img = resize_image(img, new_size=[256, 256])
        img = crop_image(img, crop_size=[224, 224])
        x.append(img)
    x = np.array(x)
    return x

#    path, ext = os.path.splitext( os.path.basename(file_paths[0]) )

def save_images_labels(saved_name, x, y, test_images, test_names):
    '''
    Saves numpy array containing train images (x), train labels (y), test images, and test image file names.
    '''
    return np.savez(saved_name, x, y, test_images, test_names)

def train_validation_split(saved_arr):
    '''
    Splits train and validation data and images.
    '''
    data = np.load(saved_arr)
    x = data['x']
    y = data['y']
    test_images = data['test_images']
    test_names = data['test_names']
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    print('X_train: {} \ny_train: {} \nX_test: {} \ny_test: {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    return X_train, X_test, y_train, y_test, test_images, test_names

# Use keras pre-trained VGG16
def cnn_model_vgg16(x_train, x_test, y_train, y_test, batch_size=22, epochs=1, input_shape=(224,224,3)):
    '''
    Builds and runs keras cnn on top of pre-trained VGG16. Data are generated from x_train.
    '''
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    add_model = Sequential()
    add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    add_model.add(Dense(256, activation='relu'))
    add_model.add(Dense(1, activation='sigmoid'))

    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
    model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])
    model.summary()

    #Data generator
    train_datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)
    train_datagen.fit(x_train)

    history = model.fit_generator(
        train_datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=x_train.shape[0] // batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[ModelCheckpoint('VGG16-transferlearning.model', monitor='val_acc', save_best_only=True)]
    )

# def predict_test_data(test_images, model):
#     '''
#     Output: model predictions
#     '''
    test_images = test_images.astype('float32')
    test_images /= 255
    predictions = model.predict(test_images)
    return predictions

def make_submission(test_labels, test_names, predictions):
    for i, name in enumerate(test_names):
        test_labels.loc[test_labels['name'] == name, 'invasive'] = predictions[i]
    test_labels.to_csv("submit.csv", index=False)


if __name__ == '__main__':
    # train_labels = load_data_labels('data/train_labels.csv')
    # train_filepaths, train_y = load_images('data/train/', train_labels)
    # train_images = process_images(train_filepaths)
    #
    # test_labels = load_data_labels('data/sample_submission.csv')
    # test_filepaths, test_y_preloaded, test_names = load_images('data/test/', test_labels, test=True)
    # test_images = process_images(test_filepaths)

    # saved_arr = save_images_labels('224.npz', x=train_images, y=train_y, test_images=test_images, test_names=test_names)
    saved_arr = '224.npz'
    X_train, X_test, y_train, y_test, test_images, test_names = train_validation_split(saved_arr)
    predictions = cnn_model_vgg16(X_train, X_test, y_train, y_test, batch_size=26, epochs=1, input_shape=(224,224,3))
    make_submission('sample_submission.csv', test_names, predictions)


    '''
    DOESNT WORK BECAUSE FILES ARE OUT OF ORDER:
    def load_images(root):
        img_dict = {}
        files = [f for f in listdir(root) if isfile(join(root, f))]
        for file_name in files:
            # if not (file_name.startswith('.')):
            img_path = '{}{}'.format(root, file_name)
            name = int(file_name.rstrip('.jpg'))
            img_cat = labels.loc[labels['name'] == name, 'invasive'].iloc[0]
            img_dict[img_path] = img_cat
        for key, value in img_dict.iteritems():
            file_paths_list.append(key)
            y.append(value)
        file_paths_list = np.array(file_paths_list)
        y = np.array(y)
        return file_paths_list, y'''
