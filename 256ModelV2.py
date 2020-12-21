# import the necessary packages

import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import numpy as np
import cv2
import sys
import os
from sklearn.metrics import confusion_matrix

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

   #spliting the char images to train and vald sets according to the split ratio and minimum pics from each category
    folderNames = ['SkyLark', 'UbuntuMono', 'SweetPuppy']
    splitRatio = 0.8

    plt.figure()
    plt.bar([f'SkyCounter {SkyCounter}', f'UbuntuCounter {UbuntuCounter}', f'SweetCounter {SweetCounter}'],
            [SkyCounter, UbuntuCounter, SweetCounter], color=['red', 'green', 'blue'])
    plt.title('Pictures per Label', fontsize=20)
    plt.xlabel('Labels', fontsize=20)
    plt.show()
    minPictures = min(SkyCounter, UbuntuCounter, SweetCounter)
    print(f'the minimum amount of pics from each category is {minPictures}')

    ### building folders for Training and validation sets
    path = 'training_data'
    if not os.path.isdir(path):
        print(f'created a new directory {path}')
        os.mkdir(path)
        for i, _ in enumerate(paths):
            os.mkdir(os.path.join(path, folderNames[i]))

    path = 'validation_data'
    if not os.path.isdir(path):
        print(f'created a new directory {path}')
        os.mkdir(path)
        for i, _ in enumerate(paths):
            os.mkdir(os.path.join(path, folderNames[i]))

    # creating a list of files for each label and shuffling them to reduce correlation between samples
    skyFiles = os.listdir(paths[0])
    ubuntuFiles = os.listdir(paths[1])
    sweetFiles = os.listdir(paths[2])
    shuffle(skyFiles)
    shuffle(ubuntuFiles)
    shuffle(sweetFiles)
    # taking only the minimum amount of pictures per label forward to the training and valid sets to balance the dataset and not overfit to the most common label!
    skyFiles = skyFiles[0:minPictures:]
    ubuntuFiles = ubuntuFiles[0:minPictures:]
    sweetFiles = sweetFiles[0:minPictures:]
    # print(len(ubuntuFiles),len(skyFiles),len(sweetFiles)) #sanity check to see we got only the min amout of pictures per label
    train_num = int(np.floor(minPictures * splitRatio))
    valid_num = int(np.floor(minPictures * (1 - splitRatio)))
    # print(train_num,valid_num) #1544 385
    i = 0
    for i in range(minPictures):
        if i < train_num:
            os.rename(os.path.join(paths[0], skyFiles[i]), f'training_data\SkyLark\\{skyFiles[i]}')
            os.rename(os.path.join(paths[1], ubuntuFiles[i]), f'training_data\\UbuntuMono\\{ubuntuFiles[i]}')
            os.rename(os.path.join(paths[2], sweetFiles[i]), f'training_data\SweetPuppy\\{sweetFiles[i]}')
        else:
            os.rename(os.path.join(paths[0], skyFiles[i]), f'validation_data\SkyLark\\{skyFiles[i]}')
            os.rename(os.path.join(paths[1], ubuntuFiles[i]), f'validation_data\\UbuntuMono\\{ubuntuFiles[i]}')
            os.rename(os.path.join(paths[2], sweetFiles[i]), f'validation_data\SweetPuppy\\{sweetFiles[i]}')



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



def train_model():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'training_data',
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="grayscale",
        batch_size=32,
        image_size=(256, 256),
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
        image_size=(256, 256),
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
                                                                  input_shape=(256,
                                                                               256,
                                                                               1)),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
            tf.keras.layers.experimental.preprocessing.RandomContrast(0.1),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )

    model = tf.keras.models.Sequential([
        data_augmentation,
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(256, 256, 1)),
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
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

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

    epochs = 120
    filepath = "256ModelV2.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                                    mode='max')
    callbacks_list = [checkpoint]
    print(model.summary())
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
        image, target_size=(256, 256), color_mode='grayscale')

    plt.imshow(img)
    plt.show()

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batc
    pred = best_model.predict_classes(img_array)
    class_names = ["Skylark", 'Sweet Puppy','Ubuntu Mono']

    print(f'the predicted label is: {class_names[int(pred)]}')
    print(f' the model is sure about it in :{best_model.predict(img_array)[0][pred]}')

def predict_9_random_picture_from_each_class():
    class_names = ["Skylark", 'Sweet Puppy','Ubuntu Mono']

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
                pics[random.randint(0, 385)], target_size=(256, 256), color_mode='grayscale')
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            pred = best_model.predict_classes(img_array)

            plt.title(
                f'the predicted label is: {class_names[int(pred)]} the model is sure about it in :{best_model.predict(img_array)[0][pred]} ')
            plt.imshow(img, cmap='gray')
        plt.show()

def outputCF(actual,pred,wordOrChar):
    cf = confusion_matrix(actual, pred)
    df_cm = pd.DataFrame(cf, ["Skylark", 'Sweet Puppy', 'Ubuntu Mono'],
                         ["Skylark", 'Sweet Puppy', 'Ubuntu Mono'])

    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.suptitle(f' Per {wordOrChar} ')
    plt.show()
def outputOccurrencePlot(wrongWords,rightWords):
    wrongOccurrence = dict()
    for word in wrongWords:
        try:
            wrongOccurrence[len(word)] = wrongOccurrence[len(word)] + 1
        except KeyError:
            wrongOccurrence[len(word)] = 1

    df = pd.DataFrame([(i, wrongOccurrence[i]) for i in wrongOccurrence.keys()],
                      columns=["length", "count"])
    fig = plt.figure(figsize=(15, 5))
    ax = sns.barplot(x="length", y="count", data=df)
    plt.suptitle('wrong predicted wrongs length')
    plt.show()

    rightOccurrence = dict()
    for word in rightWords:
        try:
            rightOccurrence[len(word)] = rightOccurrence[len(word)] + 1
        except KeyError:
            rightOccurrence[len(word)] = 1

    df = pd.DataFrame([(i, rightOccurrence[i]) for i in rightOccurrence.keys()],
                      columns=["length", "count"])
    print(df.describe())
    print(df.head())
    fig = plt.figure(figsize=(15, 5))
    ax = sns.barplot(x="length", y="count", data=df)
    plt.suptitle('right predicted wrongs length')
    plt.show()
def outputWronglyPredictedLabels(wrongCounterChars,wrongChars):
    plt.figure(figsize=(25, 10))
    plt.suptitle(f'Wrongly predicted labels')
    for i in range(0, 9):
        plt.subplot(331 + i)
        randomInt = random.randint(0, wrongCounterChars - 1)  # getting a random picture of a char from the wrong ones
        plt.title(
            f'the predicted label is: {wrongChars[randomInt][2]} , while the real label is {wrongChars[randomInt][1]} ')
        plt.imshow(wrongChars[randomInt][0], cmap='gray')
    plt.show()

def test_predict(best_model):
    if not os.path.isdir('test_data'):
        os.mkdir('test_data')
        print("created a new directory test_data, please put the file 'SynthTextTest.h5' there.")
    if not os.listdir('test_data'):
        print('the folder is empty ,please fill it according to the manual, breaking now')
        return 'error'

    best_model=best_model
    file_name = 'test_data/SynthTextTest.h5'
    db = h5py.File(file_name, 'r')
    im_names = list(db['data'].keys())
    plt.figure()
    class_names = ["Skylark", 'Sweet Puppy','Ubuntu Mono']
    wrongCounterWords=0
    rightCounterWords=0
    rightCounterChars=0
    wrongCounterChars=0
    char_actual=[]
    word_actual=[]
    char_pred=[]
    word_pred=[]
    wrongChars=[]
    wrongWords=[]
    rightWords=[]


    for im_name in im_names:
        img=db['data'][im_name][:]
        charBB = db['data'][im_name].attrs['charBB']
        txt = db['data'][im_name].attrs['txt']
        font = db['data'][im_name].attrs['font']
        #print(img)
        #print(im_name)
        #print(font)
        #print(f'the text : {txt}')
        #print(txt[0],len(txt[0])) #txt[0] is the first word with only the real word leng (not including the ' and b )
        i = 0
        for words in txt: # taking every word in the txt and parsing chars
            pics=[] #pictures of the char from the word
            for char in words:  # going through all the pictures

                pts1 = np.float32([charBB[:, :, i].T[0], charBB[:, :, i].T[1], charBB[:, :, i].T[3], charBB[:, :, i].T[2]])
                pts2 = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv.warpPerspective(img, M, (400, 400))  # cropping out a 400,400 pic of the char
                dst=cv.cvtColor(dst,cv.COLOR_BGR2GRAY)
                dst=cv.resize(dst,(256,256))
                pics.append(dst)
                #plt.imshow(dst,cmap='gray')  # showing the croped pic
                #plt.show()
                i += 1
            fontProb=[0,0,0] #Skylark", 'Sweet Puppy','Ubuntu Mono
            for j in range(len(pics)):
                img_array = tf.keras.preprocessing.image.img_to_array(pics[j])
                img_array = tf.expand_dims(img_array, 0)
                predLabel = np.argmax(best_model.predict(img_array), axis=-1)
                fontProb+=best_model.predict(img_array)[0] #each char cast a vote

                #print(f'the predicted label is: {class_names[int(predLabel)]} the model is sure about it in :{best_model.predict(img_array)[0][predLabel]} ')
            #print(fontProb)
            predictedFont=class_names[np.argmax(fontProb)] # declaring the winner of the elecetions!
            #print(f' the predicted font for the word {words} is {predictedFont} and the real label is {font[i-1]}')
            fontName=str(font[i-1])
            if predictedFont[-1] == chr(font[i-1][-1]): #an easy quick check by only using one char
                rightCounterWords+=1 #words
                rightCounterChars+=len(pics) #chars
                rightWords.append(words)
            else:
                wrongCounterWords+=1
                wrongCounterChars+=len(pics)
                wrongWords.append(words)
                for pic in pics: #gathering all the wrong prediction of chars so we can take a look at them later
                    wrongChars.append([pic,fontName[2:-1],predictedFont])

            fontName=str(font[i-1])
            for _ in range (len(pics)): # looping and inserting every char of the word
                char_actual.append(fontName[2:-1])
                char_pred.append(predictedFont)
            word_actual.append(fontName[2:-1]) #appending only full words
            word_pred.append(predictedFont)

    ########## End of Model ##########
    ########## Visualization ##########
    print(f'the model predicted {rightCounterWords} words correctly and {wrongCounterWords} wrong that {(rightCounterWords/(rightCounterWords+wrongCounterWords))*100}% accuracy!')
    print(f'the model predicted {rightCounterChars} Chars correctly and {wrongCounterChars} wrong that {(rightCounterChars/(rightCounterChars+wrongCounterChars))*100}% accuracy!')
    print(f" Class report for classifier per Words \n{metrics.classification_report(word_actual, word_pred)}")
    print(f" Class report for classifier per Chars \n{metrics.classification_report(char_actual, char_pred)}")
    print(f' the words that the model predicted wrong are : {wrongWords}')
    print(f' the words that the model predicted right are : {rightWords}')

    outputCF(char_actual,char_pred,'Char') #plotting out a nice looking confu matrix
    outputCF(word_actual,word_pred,'Word')
    outputOccurrencePlot(wrongWords,rightWords) #potting out barplot of Occurrence of right predicted words and wrong predicted words sorted by word length for additonal insights
    outputWronglyPredictedLabels(wrongCounterChars,wrongChars) # plotting out 9 pictures of wrongly predicted words




if __name__ == '__main__':

    #preprocess()
    #train_model()
    bestModelPath = '256ModelV2.hdf5'
    best_model=load_model(bestModelPath)
    #predict_9_random_picture_from_each_class()
    test_predict(best_model)


    ###########################
















# See PyCharm help at https://www.jetbrains.com/help/pycharm/
