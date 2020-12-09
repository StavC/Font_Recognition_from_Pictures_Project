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


def print_hi(name):
    print('finished')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    SkyCounter, UbuntuCounter, SweetCounter=preprocess()
    SkyCounter, UbuntuCounter, SweetCounter=1930,2974,7334
    second(SkyCounter, UbuntuCounter, SweetCounter)
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
