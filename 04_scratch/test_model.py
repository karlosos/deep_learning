from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from train_model import load_images

def test_model_noise():
    X_train, X_test, y_train, y_test = load_images()

    height, width = X_test[0].shape
    noises = np.arange(0, 100, 5)

    acc = []

    for noise in noises:
        X_test = X_test + noise * np.random.randn(height, width)

        model = load_model("model.h5")
        res = model.evaluate(X_test, y_test)
        print(f"Noise: {noise}, Res: {res}")
        acc.append(res[1])

    plt.plot(noises, acc)
    plt.xlabel("Szum")
    plt.ylabel("Dokładność")
    plt.show()


if __name__ == "__main__":
    test_model_noise()
