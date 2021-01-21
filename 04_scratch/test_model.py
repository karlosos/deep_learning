from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from train_model import load_images

def test_model_noise():
    X_train, X_test, y_train, y_test = load_images()

    height, width = X_test[0].shape
    noises = np.arange(0, 100, 5)
    # noises = np.arange(-100, 100, 5)

    acc = []

    for noise in noises:
        X_test_2 = X_test + noise * np.random.randn(height, width)
        # X_test_2 = X_test + noise

        model = load_model("model.h5")
        res = model.evaluate(X_test_2, y_test)
        print(f"Noise: {noise}, Res: {res}")
        acc.append(res[1])

    plt.plot(noises, acc)
    plt.xlabel("Szum")
    plt.ylabel("Dokładność")
    plt.show()

def test_model_brighness():
    X_train, X_test, y_train, y_test = load_images()

    height, width = X_test[0].shape
    brighnesses = np.arange(-100, 100, 5)

    acc = []

    for brighness in brighnesses:
        X_test_2 = X_test + brighness 

        model = load_model("model.h5")
        res = model.evaluate(X_test_2, y_test)
        print(f"Noise: {brighness}, Res: {res}")
        acc.append(res[1])

    plt.plot(brighnesses, acc)
    plt.xlabel("Jasność")
    plt.ylabel("Dokładność")
    plt.show()


if __name__ == "__main__":
    # test_model_noise()
    test_model_brighness()
