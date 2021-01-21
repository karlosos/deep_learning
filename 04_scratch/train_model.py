import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


def load_images():
    images = []
    labels = []
    for i in range(40):
        base_path = os.path.sep.join([".\\att_faces", f"s{i+1}"])
        for j in range(10):
            image_path = os.path.sep.join([base_path, f"{j+1}.pgm"])
            image = plt.imread(image_path)
            images.append(image)
            labels.append(i)

    X = np.array(images)
    y = np.array(labels)
    y = tf.keras.utils.to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    # X_train = X_train / 255
    # X_test = X_test / 255

    return X_train, X_test, y_train, y_test

def create_model(img_shape, filters=(32, 64, 128)):
    inputShape = (img_shape[0], img_shape[1], 1)

    chanDim = -1
        
    inputs = Input(shape=inputShape)
   
    for (i, f) in enumerate(filters):
       
        if i == 0:
            x = inputs
        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        
    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.25)(x)
    x = Dense(40)(x)
    x = Activation("softmax")(x)
    
    model = Model(inputs, x)

    opt = SGD(lr=1e-3, momentum=1e-3/200)

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
       
    return model


def main():
    X_train, X_test, y_train, y_test = load_images()
    model = create_model(X_train[0].shape)
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=True)
    res = model.evaluate(X_test, y_test)


if __name__ == "__main__":
    main()
