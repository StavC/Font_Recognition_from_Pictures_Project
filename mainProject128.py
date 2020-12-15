# set the matplotlib backend so figures can be saved in the background
# import the necessary packages
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import numpy as np
import argparse
import cv2
import sys
import os
import h5py
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from random import  shuffle
from sklearn.cluster import MiniBatchKMeans
from sklearn import svm
from sklearn import metrics
import pandas as pd
import tensorflow as tf
import random

def preprocess():


    file_name = 'font_recognition_train_set/SynthText.h5'
    db = h5py.File(file_name, 'r')
    im_names = list(db['data'].keys())


    '''
    im = im_names[0]
    img = db['data'][im][:]
    font = db['data'][im].attrs['font']
    txt = db['data'][im].attrs['txt']
    charBB = db['data'][im].attrs['charBB']
    wordBB = db['data'][im].attrs['wordBB']
    #print(img)
    #print(font)
    #print(f'char Bouding Box {charBB}')
    #print(wordBB)
    '''

    ###############################
    font_name = ["b'Skylark'", "b'Ubuntu Mono'", "b'Sweet Puppy'"]
    paths=['font_recognition_train_set\images\Skylark','font_recognition_train_set\images\\Ubuntu','font_recognition_train_set\images\Sweet']

    for path in paths:
        if not os.path.isdir(path):
            print(f'created a new directory {path}')
            os.mkdir(path)
    SkyLarkPics=[]
    UbuntuMonoPics=[]
    SweetPuppyPics=[]
    plt.figure()
    SkyCounter=0
    UbuntuCounter=0
    SweetCounter=0
    for im_name in im_names:
        img=db['data'][im_name][:]
        charBB = db['data'][im_name].attrs['charBB']
        txt = db['data'][im_name].attrs['txt']
        font = db['data'][im_name].attrs['font']

        #print(f'the text : {txt}')
        #print(font)

        i=0
        for char in font: #going through all the pictures

            pts1 = np.float32([charBB[:, :, i].T[0], charBB[:, :, i].T[1], charBB[:, :, i].T[3], charBB[:, :, i].T[2]])
            pts2 = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv.warpPerspective(img, M, (400, 400)) # cropping out a 400,400 pic of the char
            #plt.imshow(dst) #showing the croped pic
            #plt.show()
            #print(f' real {font[i]}') #priting out the real label of the font used for checkingout the code

            if str(font[i])==font_name[0]:
                #print('Skylark')
                cv2.imwrite(os.path.join(paths[0], f'Skylark{SkyCounter}.jpg'), dst)
                SkyCounter+=1
                #array_img = tf.keras.preprocessing.image.img_to_array(dst)
                #array_img=array_img/255.0
                #SkyLarkPics.append(array_img)
            elif str(font[i])==font_name[1]:
                #print('Ubuntu Mono')
                cv2.imwrite(os.path.join(paths[1], f'Ubuntu{UbuntuCounter}.jpg'), dst)
                UbuntuCounter += 1
                #array_img = tf.keras.preprocessing.image.img_to_array(dst)
                #array_img = array_img / 255.0
                #UbuntuMonoPics.append(array_img)
            else:
                cv2.imwrite(os.path.join(paths[2], f'Sweet{SweetCounter}.jpg'), dst)
                SweetCounter += 1
                #print('Sweet Puppy')
                #array_img = tf.keras.preprocessing.image.img_to_array(dst)
                #array_img = array_img / 255.0
                #SweetPuppyPics.append(array_img) #was dst
            i += 1
    print('Finished preprocessing the data, there are 3 new folders under images that represent all the chars pics sorted by fonts')
    return SkyCounter,UbuntuCounter,SweetCounter

    #font = db['data'][im_name].attrs['font']
        #txt = db['data'][im_name].attrs['txt']
        #charBB = db['data'][im_name].attrs['charBB']
        #wordBB = db['data'][im_name].attrs['wordBB']

        #plt.imshow(img)
        #plt.show()

    ###############################

    '''
    showing the pictures with bouding boxes
    font_name = ['Skylark', 'Ubuntu Mono', 'Sweet Puppy']
    nC = charBB.shape[-1]
    plt.figure()
    plt.imshow(img)
    for b_inx in range(nC):
        if font[b_inx].decode('UTF-8') == font_name[0]:
            color = 'r'
        elif font[b_inx].decode('UTF-8') == font_name[1]:
            color = 'b'
        else:
            color = 'g'
        bb = charBB[:, :, b_inx]
        x = np.append(bb[0, :], bb[0, 0])
        y = np.append(bb[1, :], bb[1, 0])
        plt.plot(x, y, color)
        # plot the word's BB:
    nW = wordBB.shape[-1]
    for b_inx in range(nW):
        bb = wordBB[:, :, b_inx]
        x = np.append(bb[0, :], bb[0, 0])
        y = np.append(bb[1, :], bb[1, 0])
    plt.plot(x, y, 'k')
    plt.show()
    '''
def second(SkyCounter, UbuntuCounter, SweetCounter):
    paths=['font_recognition_train_set\images\Skylark','font_recognition_train_set\images\\Ubuntu','font_recognition_train_set\images\Sweet']#delete when moved up
    folderNames=['SkyLark','UbuntuMono','SweetPuppy']
    splitRatio=0.8

    plt.figure()
    plt.bar([f'SkyCounter {SkyCounter}',f'UbuntuCounter {UbuntuCounter}',f'SweetCounter {SweetCounter}'],[SkyCounter,UbuntuCounter,SweetCounter],color=['red','green','blue'])
    plt.title('Pictures per Label', fontsize=20)
    plt.xlabel('Labels', fontsize=20)
    plt.show()
    minPictures = min(SkyCounter, UbuntuCounter, SweetCounter)
    print(f'the minimum amount of pics from each category is {minPictures}')


    ### building folders for Training and validation sets
    path='training_data'
    if not os.path.isdir(path):
        print(f'created a new directory {path}')
        os.mkdir(path)
        for i,_ in enumerate(paths):
            os.mkdir(os.path.join(path,folderNames[i]))

    path = 'validation_data'
    if not os.path.isdir(path):
        print(f'created a new directory {path}')
        os.mkdir(path)
        for i,_ in enumerate(paths):
            os.mkdir(os.path.join(path,folderNames[i]))

    #creating a list of files for each label and shuffling them to reduce correlation between samples
    skyFiles=os.listdir(paths[0])
    ubuntuFiles=os.listdir(paths[1])
    sweetFiles=os.listdir(paths[2])
    shuffle(skyFiles)
    shuffle(ubuntuFiles)
    shuffle(sweetFiles)
    # taking only the minimum amount of pictures per label forward to the training and valid sets to balance the dataset and not overfit to the most common label!
    skyFiles=skyFiles[0:minPictures:]
    ubuntuFiles=ubuntuFiles[0:minPictures:]
    sweetFiles=sweetFiles[0:minPictures:]
    #print(len(ubuntuFiles),len(skyFiles),len(sweetFiles)) #sanity check to see we got only the min amout of pictures per label
    train_num=int(np.floor(minPictures*splitRatio))
    valid_num=int(np.floor(minPictures*(1-splitRatio)))
    #print(train_num,valid_num) #1544 385
    i=0
    for i in range (minPictures):
        if i< train_num:
            os.rename(os.path.join(paths[0],skyFiles[i]),f'training_data\SkyLark\\{skyFiles[i]}')
            os.rename(os.path.join(paths[1],ubuntuFiles[i]),f'training_data\\UbuntuMono\\{ubuntuFiles[i]}')
            os.rename(os.path.join(paths[2],sweetFiles[i]),f'training_data\SweetPuppy\\{sweetFiles[i]}')
        else:
            os.rename(os.path.join(paths[0], skyFiles[i]), f'validation_data\SkyLark\\{skyFiles[i]}')
            os.rename(os.path.join(paths[1], ubuntuFiles[i]), f'validation_data\\UbuntuMono\\{ubuntuFiles[i]}')
            os.rename(os.path.join(paths[2], sweetFiles[i]), f'validation_data\SweetPuppy\\{sweetFiles[i]}')



def train_model():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'training_data',
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="grayscale",
        batch_size=32,
        image_size=(128, 128),
        shuffle=True,
        seed=1,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'validation_data',
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="grayscale",
        batch_size=32,
        image_size=(128, 128),
        shuffle=True,
        seed=1,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
    )

    class_names = train_ds.class_names
    classes = train_ds.class_names

    print(class_names)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"), cmap='gray')
            plt.title(class_names[i % 3])
            plt.axis("off")
    plt.show()

    data_augmentation = tf.keras.models.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal",
                                                                  input_shape=(128,
                                                                               128,
                                                                               1)),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
            tf.keras.layers.experimental.preprocessing.RandomContrast(0.1),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )

    model = tf.keras.models.Sequential([
        # This is the first convolution
        data_augmentation,
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(128, 128, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.25),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    adam = tf.keras.optimizers.Adam(lr=0.01, decay=1e-6)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    epochs = 100
    filepath = "CNNbest.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                                    mode='max')
    callbacks_list = [checkpoint]
    # callbacks=myCallback()
    history = model.fit(train_ds, batch_size=32, epochs=epochs,
                        validation_data=val_ds, callbacks=callbacks_list)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def load_model(bestModelPath):
    model = tf.keras.models.load_model(bestModelPath)
    model.summary()
    return model

def predict_one_image(path):
    image = path
    img = tf.keras.preprocessing.image.load_img(
        image, target_size=(128, 128), color_mode='grayscale')

    plt.imshow(img)
    plt.show()

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batc
    pred = best_model.predict_classes(img_array)
    class_names = ["Skylark", 'Ubuntu Mono', 'Sweet Puppy']

    print(f'the predicted label is: {class_names[int(pred)]}')
    print(f' the model is sure about it in :{best_model.predict(img_array)[0][pred]}')

def predict_9_random_picture_from_each_class():
    class_names = ["Skylark", 'Ubuntu Mono', 'Sweet Puppy']

    paths = []
    paths.append('validation_data/SkyLark')
    paths.append('validation_data/SweetPuppy')
    paths.append('validation_data/UbuntuMono')
    for i, path in enumerate(paths):
        pics = []
        for r, d, f in os.walk(path):
            for file in f:
                pics.append(os.path.join(r, file))
        plt.figure(figsize=(30, 10))
        plt.suptitle(f'the pictures were taken from {class_names[i]}  validation set')
        for i in range(0, 9):
            plt.subplot(331 + i)
            img = tf.keras.preprocessing.image.load_img(
                pics[random.randint(0, 385)], target_size=(128, 128), color_mode='grayscale')
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            pred = best_model.predict_classes(img_array)

            plt.title(
                f'the predicted label is: {class_names[int(pred)]} the model is sure about it in :{best_model.predict(img_array)[0][pred]} ')
            plt.imshow(img, cmap='gray')
        plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #SkyCounter, UbuntuCounter, SweetCounter=preprocess()
    #SkyCounter, UbuntuCounter, SweetCounter=1930,2974,7334
    #second(SkyCounter, UbuntuCounter, SweetCounter)
    #train_model()
    bestModelPath = 'CNN90.7val.hdf5'
    best_model=load_model(bestModelPath)
    predict_9_random_picture_from_each_class()


    ###########################
















# See PyCharm help at https://www.jetbrains.com/help/pycharm/
