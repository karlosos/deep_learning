# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 23:25:16 2021

@author: pawel
"""

# import 
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import os


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import concatenate

import locale


def load_attributes(inputPath):

    cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
    df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)

    zipcodes = df["zipcode"].value_counts().keys().tolist()
    counts = df["zipcode"].value_counts().tolist()
    
    for (zipcode, count) in zip(zipcodes, counts):
        
        if count < 20:
            idxs = df[df["zipcode"] == zipcode].index
            df.drop(idxs, inplace=True)

    return df

def process_attributes(df, train, test):
   
    continuous = ["bedrooms", "bathrooms", "area"]
    
    cs = MinMaxScaler()
    trainContinuous = cs.fit_transform(train[continuous])
    testContinuous = cs.transform(test[continuous])
    
    zipBinarizer = LabelBinarizer().fit(df["zipcode"])
    trainCategorical = zipBinarizer.transform(train["zipcode"])
    testCategorical = zipBinarizer.transform(test["zipcode"])
   
    trainX = np.hstack([trainCategorical, trainContinuous])
    testX = np.hstack([testCategorical,
                    testContinuous])
    
    return (trainX, testX)





inputPath = 'houses/HousesInfo.txt'
df = load_attributes(inputPath)


def scale(im, nR, nC):
    nR0 = len(im)     # source number of rows 
    nC0 = len(im[0])  # source number of columns 
    return [[ im[int(nR0 * r / nR)][int(nC0 * c / nC)]  
             for c in range(nC)] for r in range(nR)]

def load_images(df, inputPath):
   
    images = []
    
    for i in df.index.values:
        
        basePath = os.path.sep.join([inputPath, "{}_*".format(i + 1)])
        housePaths = sorted(list(glob.glob(basePath)))
        
       
        inputImages = []
        outputImage = np.zeros((64, 64, 3), dtype="uint8")
       
        for housePath in housePaths:
           
            image = plt.imread(housePath)
           
            image = scale(image,32,32)
            inputImages.append(image)
      
        outputImage[0:32, 0:32] = inputImages[0]
        outputImage[0:32, 32:64] = inputImages[1]
        outputImage[32:64, 32:64] = inputImages[2]
        outputImage[32:64, 0:32] = inputImages[3]
       
        images.append(outputImage)
  
    return np.array(images)

def create_cnn(width, height, depth, filters=(16, 32, 64)):
   
    inputShape = (height, width, depth)
    chanDim = -1
    
    inputs = Input(shape=inputShape)
   
    for (i, f) in enumerate(filters):
       
        if i == 0:
            x = inputs
        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
   
    x = Dense(4)(x)
    x = Activation("relu")(x)
    
    model = Model(inputs, x)
   
    return model


images = load_images(df, 'houses')
images = images / 255.0


split = train_test_split(df, images, test_size=0.25, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split

maxPrice = trainAttrX["price"].max()
trainY = trainAttrX["price"] / maxPrice
testY = testAttrX["price"] / maxPrice



(trainAttrX, testAttrX) = process_attributes(df,
	trainAttrX, testAttrX)


trainY = np.array(trainY)
testY = np.array(testY)
trainAttrX = np.array(trainAttrX)
testAttrX = np.array(testAttrX)


cnn = create_cnn(64, 64, 3)

x = Dense(4, activation="relu")(cnn.output)
x = Dense(1, activation="linear")(x)

model = Model(inputs=[ cnn.input], outputs=x)


opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)


model.fit(
	x=[ trainImagesX], y=trainY,
	validation_data=([testImagesX], testY),
	epochs=100, batch_size=8)


preds = model.predict([testImagesX])




diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

mean = np.mean(absPercentDiff)


print("srednia cena: {}".format(df["price"].mean(), grouping=True))
	
print("srednia błąd: {:.2f}%".format(mean))
