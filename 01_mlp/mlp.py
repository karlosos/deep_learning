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


def run_test(opt, epochs=10):
    trainX, trainY, testX, testY = load_data()

    model = create_model(opt=opt, activation="sigmoid")

    model.fit(trainX, trainY, epochs=epochs, batch_size=256, verbose=1)
    model.save("mlp.h5")
    res = model.evaluate(testX, (testY))
    return res


def main():
    """
    zbadać wpływ funkcji aktywacji (activation=”..”) na jakość uczenia - sprawdzić funkcje aktywacji: sigmoid, hard_sigmoid, tanh, linear, relu, softmax
    zbadać wpływ liczby epok uczenia na jakość klasyfikacji np. {10,100,1000}
    zbadać wpływ optymalizatora {adam,sgd,adadelta,adagrad,rmsprop} na jakość uczenia
    zbadać wpływ kroku uczenia (learning rate) na jakość uczenia
    """

    learning_rates = [0.05, 0.01, 0.02, 0.1]
    optimizers = [
        tf.keras.optimizers.SGD,
        tf.keras.optimizers.Adam,
        tf.keras.optimizers.Adadelta,
        tf.keras.optimizers.Adagrad,
        tf.keras.optimizers.RMSprop,
    ]
    data = {"optimizer": [], "lr": [], "loss": [], "acc": []}

    for lr in learning_rates:
        for optimizer in optimizers:
            opt = optimizer(lr=lr)
            res = run_test(epochs=10, opt=opt)

            data["optimizer"].append(opt)
            data["lr"].append(lr)
            data["loss"].append(res[0])
            data["acc"].append(res[1])

    df = pd.DataFrame.from_dict(data)
    print(df)


if __name__ == "__main__":
    main()
