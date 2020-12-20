from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np


def load_image(file_path):
    img = load_img(file_path, color_mode="grayscale", target_size=(28, 28))

    # Rotacja
    img = img.rotate(angle=30)

    # Translacja
    img = translate(img, -2, axis=0)
    
    # Szum
    img = img + 10 * np.random.randn(28, 28)

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
    img = load_image("sample_image.png")
    model = load_model("models/mlp_100_Adam_0.02_sigmoid.h5")
    digit = model.predict_classes(img)
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.title(f"Predykcja: {digit[0]}")
    plt.show()


if __name__ == "__main__":
    main()
