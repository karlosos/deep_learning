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



def create_mlp(dim):
    
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu"))
    model.add(Dense(4, activation="relu"))
   
    return model



inputPath = 'houses/HousesInfo.txt'
df = load_attributes(inputPath)



split = train_test_split(df, test_size=0.25, random_state=42)
(trainAttrX, testAttrX) = split

maxPrice = trainAttrX["price"].max()
trainY = trainAttrX["price"] / maxPrice
testY = testAttrX["price"] / maxPrice


(trainAttrX, testAttrX) = process_attributes(df,
	trainAttrX, testAttrX)


trainY = np.array(trainY)
testY = np.array(testY)
trainAttrX = np.array(trainAttrX)
testAttrX = np.array(testAttrX)


mlp = create_mlp(trainAttrX.shape[1])

x = Dense(1, activation="linear")(mlp.output)

model = Model(inputs=[mlp.input], outputs=x)


opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

model.fit(
	x=[trainAttrX], y=trainY,
	validation_data=([testAttrX], testY),
	epochs=100, batch_size=8)


preds = model.predict([testAttrX])



diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

mean = np.mean(absPercentDiff)


print("srednia cena: {}".format(df["price"].mean(), grouping=True))
	
print("srednia błąd: {:.2f}%".format(mean))
