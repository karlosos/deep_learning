import tensorflow as tf
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


def create_model(opt, activation="sigmoid"):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten())
    model.add(
        tf.keras.layers.Dense(10, activation=activation, input_shape=(28 * 28, 1))
    )

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def run_test(epochs, optimizer, lr, activation):
    trainX, trainY, testX, testY = load_data()

    opt = optimizer(lr=lr)
    model = create_model(opt=opt, activation=activation)

    model.fit(trainX, trainY, epochs=epochs, batch_size=256, verbose=1)
    model.save(f"./models/mlp_{epochs}_{opt.__class__.__name__}_{lr}_{activation}.h5")
    res = model.evaluate(testX, (testY))
    return res


def experiment_1():
    """
    zbadać wpływ funkcji aktywacji (activation=”..”) na jakość uczenia
    sprawdzić funkcje aktywacji: sigmoid, hard_sigmoid, tanh, linear, relu, softmax
    zbadać wpływ optymalizatora {adam,sgd,adadelta,adagrad,rmsprop} na jakość uczenia
    zbadać wpływ kroku uczenia (learning rate) na jakość uczenia
    """

    data = {
        "optimizer": [],
        "epochs": [],
        "lr": [],
        "activation": [],
        "loss": [],
        "acc": [],
    }

    learning_rates = [0.05, 0.01, 0.02, 0.1]
    optimizers = [
        tf.keras.optimizers.SGD,
        tf.keras.optimizers.Adam,
        tf.keras.optimizers.Adadelta,
        tf.keras.optimizers.Adagrad,
        tf.keras.optimizers.RMSprop,
    ]
    activations = ["sigmoid", "hard_sigmoid", "tanh", "linear", "relu", "softmax"]
    epochs = [10, 100]

    for e in epochs:
        for lr in learning_rates:
            for optimizer in optimizers:
                for activation in activations:
                    res = run_test(
                        epochs=e, optimizer=optimizer, lr=lr, activation=activation
                    )

                    data["optimizer"].append(optimizer.__name__)
                    data["epochs"].append(e)
                    data["lr"].append(lr)
                    data["activation"].append(activation)
                    data["loss"].append(res[0])
                    data["acc"].append(res[1])

    df = pd.DataFrame.from_dict(data)
    print(df)
    df.to_csv("experiment_1.csv", index=False)


def experiment_2():
    """
    Zbadać wpływ liczby epok uczenia na jakość klasyfikacji np. {10,100,1000}

    Badania wpływ epok na modelu:
        * Adagrad
        * LR = 0.10
        * Sigmoid
    """
    lr = 0.01
    optimizer = tf.keras.optimizers.Adagrad
    epochs = 1000
    activation = "sigmoid"

    trainX, trainY, testX, testY = load_data()

    opt = optimizer(lr=lr)
    model = create_model(opt=opt, activation=activation)

    csv_logger = tf.keras.callbacks.CSVLogger(
        f"experiments/epochs_experiment_{lr}.csv", append=True
    )

    model.fit(
        trainX, trainY, epochs=epochs, batch_size=256, verbose=1, callbacks=[csv_logger]
    )
    model.save(f"./models/mlp_{epochs}_{optimizer.__name__}_{lr}_{activation}.h5")
    res = model.evaluate(testX, (testY))

    print(res)


if __name__ == "__main__":
    import time

    t1 = time.time()
    # experiment_1()
    experiment_2()
    print("Time:", time.time() - t1)
