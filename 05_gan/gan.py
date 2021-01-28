# -*- coding: utf-8 -*-
"""

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


train_images=np.zeros((LICZBA_OBRAZOW_BAZOWYCH,BOX_SIZE,BOX_SIZE))

for i in range(0,LICZBA_OBRAZOW_BAZOWYCH):
    #wczytanie obrazu
	im=plt.imread(...)
	#niezbędy pre-processing (grayscale, skalowanie)
    
	train_images[i-1,:,:]=im
    
train_images = train_images.reshape(train_images.shape[0], BOX_SIZE, BOX_SIZE, 1).astype('float32')
#normalizacja względem zera (-1,1)
train_images = (train_images - 127.5) / 127.5 


#mieszanie zbioru uczącego
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

#budowa modelu sieci generatora, na wyjściu warstwa o wymiarach obrazu
# można uzyc warstw laysers.Dense, layers.BatchNormalization, layers.LeakyReLU, layers.Conv2DTranspose

def gen_model():
    model = tf.keras.Sequential()
#...
    return model

#utworzenie instancji modelu

generator = gen_model()


#budowa modelu sieci generatora, na wyjściu warstwa do klasyfikacji binarnej
# można uzyc warstw laysers.Conv2D, layers.LeakyReLU, layers.Dropout, layers.Flatten + layers.Dense-na końcu 1 neuron 0-1

def disc_model():
    model = tf.keras.Sequential()
    #...
    return model

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

@parametry uczenia generatora i dyskryminatora
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
