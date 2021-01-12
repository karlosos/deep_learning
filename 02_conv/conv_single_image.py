from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np


def load_image(file_path, translation=0, rotation=0, noise=0):
    img = load_img(file_path, color_mode="grayscale", target_size=(28, 28))

    # Rotacja
    img = img.rotate(angle=rotation)

    # # Translacja
    img = translate(img, translation, axis=0)

    # Szum
    img = img + noise * np.random.randn(28, 28)

    #  Konwersja do macierzy
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype("float32")
    img = img / 255.0

    return img


def translate(img, value, axis):
    img = np.roll(img, value, axis=axis)
    return img


def main():
    img = load_image("sample_image.png", translation=0, rotation=30, noise=0)
    model = load_model("models/conv_2_1000_0.02.h5")
    digit = model.predict_classes(img)
    plt.imshow(img.reshape(28, 28), cmap="gray")
    plt.title(f"Predykcja: {digit[0]}")
    plt.show()


if __name__ == "__main__":
    main()
