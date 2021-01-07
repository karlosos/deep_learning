import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import Flatten 
from tensorflow.keras.optimizers import SGD 
import pandas as pd


def load_data():
    # load data
    (trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()

    # tailor data to classification problem
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    trainY = tf.keras.utils.to_categorical(trainY)

    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    testY = tf.keras.utils.to_categorical(testY)

    # normalization
    trainX_norm = trainX.astype("float32")
    trainX_norm = trainX_norm / 255.0

    testX_norm = testX.astype("float32")
    testX_norm = testX_norm / 255.0

    return trainX_norm, trainY, testX_norm, testY


def create_model(lr, architecture=1):
    model = tf.keras.models.Sequential()

    if architecture == 1:
        model.add(
            Conv2D(
                128,
                (3, 3),
                activation="relu",
                kernel_initializer="he_uniform",
                input_shape=(28, 28, 1),
            )
        )
        model.add(MaxPooling2D((2, 2)))
    elif architecture == 2:
        model.add(
            Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform")
        )
        model.add(MaxPooling2D((2, 2)))
        model.add(
            Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform")
        )
        model.add(MaxPooling2D((2, 2)))
    elif architecture == 3:
        model.add(
            Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform")
        )
        model.add(MaxPooling2D((2, 2)))
        model.add(
            Conv2D(16, (3, 3), activation="relu", kernel_initializer="he_uniform")
        )
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(8, (3, 3), activation="relu", kernel_initializer="he_uniform"))
        model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(10, activation="softmax"))

    opt = SGD(lr=lr, momentum=0.9)

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def run_test(epochs, lr, architecture=1):
    trainX, trainY, testX, testY = load_data()

    model = create_model(lr=lr, architecture=architecture)

    model.fit(trainX, trainY, epochs=epochs, batch_size=256, verbose=1)
    model.save(f"./models/conv_{architecture}_{epochs}_{lr}.h5")
    res = model.evaluate(testX, (testY))
    return res


def experiment_1():
    """
    Porównanie architektur i kroku uczenia
    """
    data = {
        "architecture": [],
        "lr": [],
        "loss": [],
        "acc": [],
    }

    architectures = [1, 2, 3]
    learning_rates = [0.05, 0.01, 0.02, 0.1]

    for architecture in architectures:
        for lr in learning_rates:
            res = run_test(
                epochs=10, lr=lr, architecture=architecture 
            )

            data["architecture"].append(architecture)
            data["lr"].append(lr)
            data["loss"].append(res[0])
            data["acc"].append(res[1])

    df = pd.DataFrame.from_dict(data)
    print(df)
    df.to_csv("experiment_1.csv", index=False)


def experiment_2():
    """
    Zbadać wpływ liczby epok uczenia na jakość klasyfikacji np. {10,100,1000}

    Badania wpływ epok na modelu:
        - TODO: wybrac najlepszy model z kazdej architektury 
    """
    lr = 0.01
    architecture = 1
    epochs = 1000

    trainX, trainY, testX, testY = load_data()

    opt = optimizer(lr=lr)
    model = create_model(opt=opt, activation=activation)

    csv_logger = tf.keras.callbacks.CSVLogger(
        f"experiments/epochs_experiment_{architecture}_{lr}.csv", append=True
    )

    model.fit(
        trainX, trainY, epochs=epochs, batch_size=256, verbose=1, callbacks=[csv_logger]
    )

    model.save(f"./models/conv_{architecture}_{epochs}_{lr}.h5")
    res = model.evaluate(testX, (testY))

    print(res)


if __name__ == "__main__":
    import time

    t1 = time.time()
    experiment_1()
    # experiment_2()
    print("Time:", time.time() - t1)
