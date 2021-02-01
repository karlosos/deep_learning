# https://analyticsindiamag.com/hands-on-guide-to-deep-convolutional-gan-for-fashion-apparel-image-generation/


import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras import layers
import time

#uwaga1: jeśli rozdzielczość obrazów jest zbyt wysoka (np. nie da się załadować obrazów do pamięci karty), zmniejszyć do rozmiaru 'sensownego', np. 100x100px)

# LICZBA_OBRAZOW_BAZOWYCH - zależy od wielkości zbioru
# BOX_SIZE - wielkośc pojedynczego obrazu

LICZBA_OBRAZOW_BAZOWYCH = 0
BOX_SIZE = 0

#liczba przykladow do generowania
przyklady = 12

#parametry uczenia, zmienić, jeśli za mało pamięci
BUFFER_SIZE = 60000
BATCH_SIZE = 256

# wielkośc wektora kodowego
NOISE_SIZE=100

# liczba epok uczenia
EPOCHS = 400


# ziarno do generatora, aby wizualizowac wyniki
seed = tf.random.normal([przyklady, NOISE_SIZE])




#mieszanie zbioru uczącego
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

#budowa modelu sieci generatora, na wyjściu warstwa o wymiarach obrazu
# można uzyc warstw laysers.Dense, layers.BatchNormalization, layers.LeakyReLU, layers.Conv2DTranspose

# https://towardsdatascience.com/generative-adversarial-network-gan-for-dummies-a-step-by-step-tutorial-fdefff170391
def gen_model(start_filters, filter_size, input_shape):
    # def add_generator_block(x, filters, filter_size):
    #     x = Deconvolution2D(filters, filter_size, strides=2, padding='same')(x)
    #     x = BatchNormalization()(x)
    #     x = LeakyReLU(0.3)(x)
    #     return x

    # inputs = Input(shape=input_shape)
    # x = Dense(4 * 4 * (start_filters * 8), input_dim=input_shape)(inputs)
    # x = BatchNormalization()(x)
    # x = Reshape(target_shape(4, 4, start_filters * 8))(x)

    # x = add_generator_block(x, start_filters * 4, filter_size)
    # x = add_generator_block(x, start_filters * 2, filter_size)
    # x = add_generator_block(x, start_filters, filter_size)
    # x = add_generator_block(x, start_filters, filter_size)

    # x = Conv2D(3, kernel_size=5, padding='same', activation='tanh')(x)

    # return Model(inputs=inputs, outputs=x)
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model
#utworzenie instancji modelu

generator = gen_model()


#budowa modelu sieci generatora, na wyjściu warstwa do klasyfikacji binarnej
# można uzyc warstw laysers.Conv2D, layers.LeakyReLU, layers.Dropout, layers.Flatten + layers.Dense-na końcu 1 neuron 0-1

def disc_model(start_filters, input_shape, filter_size):
    # def add_discriminator_block(x, filters, filter_size):
    #     x = Conv2D(filters, filter_size, padding='same')(x)
    #     x = BatchNormalization()(x)
    #     x = Conv2D(filters, filter_size, padding='same', strides=2)(x)
    #     x = BatchNormalization()(x)
    #     x = LeakyReLU(0.3)(x)
    #     return x

    # inputs = Input(shape=input_shape)

    # x = add_discriminator_block(inp, start_filters, filter_size)
    # x = add_discriminator_block(x, start_filters * 2, filter_size)
    # x = add_discriminator_block(x, start_filters * 4, filter_size)
    # x = add_discriminator_block(x, start_filters * 8, filter_size)

    # x = GlobalAveragePooling2D()
    # x = Dense(1, activation='sigmoid')(x)
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model   # return Model(inputs=inputs, outputs=x) 

discriminator = disc_model()


# funkcja entropi wzajemnej
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# funkcja straty dla dyskryminatora
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

#funckja straty dla generatora
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

#parametry uczenia generatora i dyskryminatora
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# jeśli chcemy przerywać uczenie i je potem kontynuować, warto zapisywać pliki ckpt
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)






# przyspieszenie obliczeń

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_SIZE])
	
#to pozwala na obserwowanie postępów uczenia i modyfikowanie gradientów

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

#podstawowa pętla uczenia    
def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    #generujemy obrazu
    generate_images(generator,
                             epoch + 1,
                             seed)

    # zapisujemy model co 15 epok
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Czas epoki {} = {} sekund'.format(epoch + 1, time.time()-start))


def generate_images(model, epoch, test_input):
  # `training` ustawiamy na False.
  # aby uruchomic wnioskowanie/generowanie
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(3,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.show()


#uruchom trening
train(train_dataset, EPOCHS)
