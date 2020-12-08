# set the matplotlib backend so figures can be saved in the background
import matplotlib
# import the necessary packages

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
from sklearn.cluster import MiniBatchKMeans
from sklearn import svm
from sklearn import metrics
import seaborn as sn
import pandas as pd

def preprocess():


    file_name = 'font_recognition_train_set/SynthText.h5'
    db = h5py.File(file_name, 'r')
    im_names = list(db['data'].keys())
    im = im_names[0]
    img = db['data'][im][:]
    font = db['data'][im].attrs['font']
    txt = db['data'][im].attrs['txt']
    charBB = db['data'][im].attrs['charBB']
    wordBB = db['data'][im].attrs['wordBB']
    print(img)
    print(font)
    print(f'the text : {txt}')
    print(f'char Bouding Box {charBB}')
    print(wordBB)

    ###############################
    plt.figure()
    for im_name in im_names:
        img=db['data'][im_name][:]
        plt.imshow(img)
        plt.show()


    ##############################
    pts1 = np.float32([charBB[:, :, 1].T[0], charBB[:, :, 1].T[1], charBB[:, :, 1].T[3], charBB[:, :, 1].T[2]])
    pts2 = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(img, M, (400, 400))
    plt.imshow(dst)
    plt.show()

    pts1 = np.float32([charBB[:, :, 2].T[0], charBB[:, :, 2].T[1], charBB[:, :, 2].T[3], charBB[:, :, 2].T[2]])
    pts2 = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(img, M, (400, 400))
    plt.imshow(dst)
    plt.show()

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

def print_hi(name):
    print('finished')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    preprocess()
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
