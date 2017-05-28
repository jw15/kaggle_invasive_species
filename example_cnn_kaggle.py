'''
example cnn from kaggle
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import math
from glob import glob
import os
from multiprocessing import Pool, cpu_count

from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint


master = pd.read_csv('data/train_labels.csv')

img_path = "data/train/"

y = []
file_paths = []
for i in range(len(master)):
    file_paths.append( img_path + str(master.ix[i][0]) +'.jpg' )
    y.append(master.ix[i][1])
y = np.array(y)

#image reseize & centering & crop

def centering_image(img):
    size = [256,256]
    img_size = img.shape[:2]
    # centering
    row = (size[1] - img_size[0]) // 2
    col = (size[0] - img_size[1]) // 2
    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)
    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img
    return resized

# def parallel_centering_image(img):
#     pool = Pool(processes=cpu_count())
#     resized = pool.map(centering_image, img)
#     return resized

# def image_processing(file_paths):
#     x = []
# #     for file_path in file_paths:
#         #read image
#     img = cv2.imread(file_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     print('original image shape: {}'.format(img.shape))
#
#     #resize
#     if(img.shape[0] > img.shape[1]):
#         tile_size = (int(img.shape[1]*256/img.shape[0]),256)
#     else:
#         tile_size = (256, int(img.shape[0]*256/img.shape[1]))
#     #centering
#     resized_img = cv2.resize(img, (tile_size[1], tile_size[0]))
#     print('resized img shape: {}'.format(resized_img.shape))
#     img = centering_image(resized_img)
# #         print('tile size before centering: {}'.format(tile_size))
# #         img = cv2.resize(img, (tile_size[1], tile_size[0]))
# #         print('img shape before sending to fxn: {}'.format(img.shape[:2]))
# #         img = parallel_centering_image(img)
#
# #         out put 224*224px
#     img = img[16:240, 16:240]
# #     print('img_shape_after_224x224 output: {}'.format(img.shape))
#     x.append(img)
#
#     x = np.array(x)
#
# def parallel_image_processing(file_paths):
#     pool = Pool(processes=cpu_count())
#     resizeds = pool.map(image_processing, file_paths)
#     return resizeds

x = []
for i, file_path in enumerate(file_paths):
    #read image
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #resize
    if(img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1]*256/img.shape[0]),256)
    else:
        tile_size = (256, int(img.shape[0]*256/img.shape[1]))

    #centering
    img = centering_image(cv2.resize(img, dsize=tile_size))

    #out put 224*224px
    img = img[16:240, 16:240]
    x.append(img)

x = np.array(x)

sample_submission = pd.read_csv("data/sample_submission.csv")
img_path = "data/test/"

test_names = []
file_paths = []

for i in range(len(sample_submission)):
    test_names.append(sample_submission.ix[i][0])
    file_paths.append( img_path + str(int(sample_submission.ix[i][0])) +'.jpg' )

test_names = np.array(test_names)

test_images = []
for file_path in file_paths:
    #read image
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #resize
    if(img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1]*256/img.shape[0]),256)
    else:
        tile_size = (256, int(img.shape[0]*256/img.shape[1]))

    #centering
    img = centering_image(cv2.resize(img, dsize=tile_size))

    #out put 224*224px
    img = img[16:240, 16:240]
    test_images.append(img)

    path, ext = os.path.splitext( os.path.basename(file_paths[0]) )

test_images = np.array(test_images)

# save numpy array
np.savez('224.npz', x=x, y=y, test_images=test_images, test_names=test_names)

# Split train and validation data

data_num = len(y)
random_index = np.random.permutation(data_num)

x_shuffle = []
y_shuffle = []
for i in range(data_num):
    x_shuffle.append(x[random_index[i]])
    y_shuffle.append(y[random_index[i]])

x = np.array(x_shuffle)
y = np.array(y_shuffle)

val_split_num = int(round(0.2*len(y)))
x_train = x[val_split_num:]
y_train = y[val_split_num:]
x_test = x[:val_split_num]
y_test = y[:val_split_num]

print('x_train', x_train.shape)
print('y_train', y_train.shape)
print('x_test', x_test.shape)
print('y_test', y_test.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# User keras pre-trained VGG16

img_rows, img_cols, img_channel = 224, 224, 3

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))

add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(1, activation='sigmoid'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

model.summary()

# Run actual model:
batch_size = 32
epochs = 50

# def train_model(x_train):
train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
train_datagen.fit(x_train)

# def train_datagen_parallel(x_train):
#     pool = Pool(processes=cpu_count())
#     return pool.map(train_model, x_train)




history = model.fit_generator(
    train_datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=x_train.shape[0] // batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=[ModelCheckpoint('VGG16-transferlearning.model', monitor='val_acc', save_best_only=True)]
)

# predict test data

test_images = test_images.astype('float32')
test_images /= 255

predictions = model.predict(test_images)

sample_submission = pd.read_csv("../input/sample_submission.csv")

for i, name in enumerate(test_names):
    sample_submission.loc[sample_submission['name'] == name, 'invasive'] = predictions[i]

sample_submission.to_csv("submit.csv", index=False)
