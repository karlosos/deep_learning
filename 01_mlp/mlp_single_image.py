from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model


def load_image(file_path):
    img = load_img(file_path, color_mode="grayscale", target_size=(28, 28))

    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype("float32")
    img = img / 255.0
    return img


def main():
    img = load_image("sample_image.png")
    model = load_model("mlp.h5")
    digit = model.predict_classes(img)
    print(digit)


if __name__ == "__main__":
    main()
