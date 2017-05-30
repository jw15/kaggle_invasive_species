'''
Adapted from code posted by user fujisan on kaggle.com, accessed at: https://www.kaggle.com/fujisan/use-keras-pre-trained-vgg16-acc-98/versions
Uses pre-trained VGG16 network ('Very Deep Convolutional Networks for Large-Scale Image Recognition.' Simonyan & Zisserman. arXiv:1409.1556).
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
np.random.seed(1337)  # for reproducibility

# Runs code on GPU
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
    if test:
        return file_paths_list, y, test_names
    else:
        return file_paths_list, y

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

def resize_image_to_square(img, new_size=[512, 512]):
    '''
    Resizes images without changing aspect ratio. Centers image in square black box.
    Input: Image, desired new size (new_size = [height, width]))
    Output: Resized image, centered in black box with dimensions new_size
    '''
    if(img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1]*new_size[1]/img.shape[0]),new_size[1])
    else:
        tile_size = (new_size[1], int(img.shape[0]*new_size[1]/img.shape[1]))
    return _center_image(cv2.resize(img, dsize=tile_size), new_size)

def crop_image(img, crop_size):
    '''
    Crops image to new_dims, centering image in frame.
    Input: Image, desired cropped size (crop_size=[height, width])
    Output: Cropped image
    '''
    row_buffer = (img.shape[0] - crop_size[0]) // 2
    col_buffer = (img.shape[1] - crop_size[1]) // 2
    return img[row_buffer:(img.shape[0] - row_buffer), col_buffer:(img.shape[1] - col_buffer)]

def process_images(file_paths_list, resize_new_size=[512,512], crop_size=[448, 448]):
    '''
    Input: list of file paths (images)
    Output: numpy array of processed images (color shifted - RGB reversed back (cv2 reverses on imread), normalized, resized, centered)
    '''
    x = []
    for file_path in file_paths_list:
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = resize_image_to_square(img, new_size=resize_new_size)
        img = crop_image(img, crop_size=crop_size)
        x.append(img)
    x = np.array(x)
    return x

def train_validation_split(saved_arr):
    '''
    Splits train and validation data and images. Also loads test images, names from saved array.
    Input: saved numpy array, files/columns in that array
    Output: Train/validation data (e.g., X_train, X_test, y_train, y_test), test images, test image names (file names minus '.png')
    '''
    data = np.load('448.npz')

    x = data.files[0]
    x = data[x]
    y = data.files[1]
    y = data[y]
    test_images = data.files[2]
    test_images = data[test_images]
    test_names = data.files[3]
    test_names = data[test_names]
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    print('X_train: {} \ny_train: {} \nX_test: {} \ny_test: {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train/255
    X_test = X_test/255
    return X_train, X_test, y_train, y_test, test_images, test_names

def cnn_model_vgg16(x_train, x_test, y_train, y_test, batch_size=22, epochs=1, input_shape=(224,224,3)):
    '''
    Builds and runs keras cnn on top of pre-trained VGG16. Data are generated from X_train.
    '''
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    add_model = Sequential()
    add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    add_model.add(Dense(512, activation='relu'))
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

    t_images = test_images.astype('float32')
    t_images /= 255
    predictions = model.predict(t_images)
    return predictions

def make_submission(sample_submission, test_names, predictions):
    for i, name in enumerate(test_names):
        sample_submission.loc[sample_submission['name'] == name, 'invasive'] = predictions[i]
    sample_submission.to_csv("submit.csv", index=False)


if __name__ == '__main__':
    train_labels = load_data_labels('train_labels.csv')
    train_filepaths, train_y = load_images('datafiles/train/train/', train_labels)
    train_images = process_images(train_filepaths, resize_new_size=[512,512], crop_size=[448, 448])

    test_labels = load_data_labels('sample_submission.csv')
    test_filepaths, test_y_preloaded, test_names = load_images('datafiles/test/', test_labels, test=True)
    test_images = process_images(test_filepaths, resize_new_size=[512,512], crop_size=[448, 448])
    np.savez('448.npz', train_images, train_y, test_images, test_names)
    X_train, X_test, y_train, y_test, test_images, test_names = train_validation_split('448.npz')
    predictions = cnn_model_vgg16(X_train, X_test, y_train, y_test, batch_size=5, epochs=40, input_shape=(448,448,3))
    sample_submission = pd.read_csv("sample_submission.csv")
    make_submission(sample_submission, test_names, predictions)
