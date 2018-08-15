# -*- coding: utf-8 -*-
"""
Train an Auxiliary Classifier Generative Adversarial Network (ACGAN) on the
MNIST dataset. See https://arxiv.org/abs/1610.09585 for more details.
"""

from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
from PIL import Image
from six.moves import range
import numpy as np
from keras.datasets import mnist
from keras import layers
from keras.layers import Input, Conv2DTranspose, Dense, Reshape, BatchNormalization, Embedding, Flatten, Conv2D, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam

def build_generator(latent_size, num_classes):
    """Build the generator model.
       Model input:
         latent: noise vector of shape (latent_size, )
         image_class: image class, int32
       Model output: 
         fake images of shape (28, 28, 1).
    """
    # random noise vector z from the AC-GAN paper
    latent = Input(shape=(latent_size, ))

    # class label c from the AC-GAN paper
    image_class = Input(shape=(1,), dtype='int32')

    emb = Embedding(num_classes, latent_size, embeddings_initializer='glorot_normal')(image_class)
    cls_vec = Flatten()(emb) # a class conditional vector of shape (latent_size,)

    # element-wise multiplication between z and the class conditional vector
    x = layers.multiply([latent, cls_vec]) # output has shape (latent_size,)

    x = Dense(3 * 3 * 384, activation='relu')(x)
    x = Reshape((3, 3, 384))(x)

    # upsample to (7, 7, 192)
    x = Conv2DTranspose(192, 5, strides=1, padding='valid', activation='relu', kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)

    # upsample to (14, 14, 96)
    x = Conv2DTranspose(96, 5, strides=2, padding='same', activation='relu', kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)

    # upsample to (28, 28, 1))
    x = Conv2DTranspose(1, 5, strides=2, padding='same', activation='tanh', kernel_initializer='glorot_normal')(x)
    fake_image = BatchNormalization()(x)

    return Model([latent, image_class], fake_image)


def build_discriminator(num_classes):
    """Build the discriminator model.
       Model input:
         image of shape (28, 28, 1)
       Model output:
         is_real probability
         class distribution
    """
    image = Input(shape=(28, 28, 1))

    x = Conv2D(32, 3, padding='same', strides=2)(image)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, 3, padding='same', strides=1)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, 3, padding='same', strides=2)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, 3, padding='same', strides=1)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    features = Flatten()(x)
    is_real = Dense(1, activation='sigmoid')(features)
    image_class = Dense(num_classes, activation='softmax')(features)

    return Model(image, [is_real, image_class])


if __name__ == '__main__':

    latent_size = 100
    num_classes = 10
    batch_size = 100
    epochs = 15

    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # build the discriminator model
    discriminator = build_discriminator(num_classes)
    discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )

    # build a combined generator/discriminator model that will
    # be used to train the generator
    generator = build_generator(latent_size, num_classes)

    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(1,), dtype='int32')
    fake_image = generator([latent, image_class])

    # The combined model should only train the generator but not the
    # discriminator. Disable training only affects the combined model 
    # and not the discriminator model because this has already been 
    # compiled.
    discriminator.trainable = False
    is_real, aux = discriminator(fake_image)
    combined = Model([latent, image_class], [is_real, aux])

    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # normalize images to range [-1,1] and add channel dim
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = (x_test.astype(np.float32) - 127.5) / 127.5
    x_test = np.expand_dims(x_test, axis=-1)

    num_train, num_test = x_train.shape[0], x_test.shape[0]

    print('x_train shape: %s' % str(x_train.shape))
    print('x_test shape: %s' % str(x_test.shape))

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    for epoch in range(1, epochs + 1):
        print('Epoch {}/{}'.format(epoch, epochs))

        num_batches = int(x_train.shape[0] / batch_size)
        
        # we don't want the discriminator to also maximize the classification
        # accuracy of the auxiliary classifier on generated images, so we
        # don't train discriminator to produce class labels for generated
        # images (see https://openreview.net/forum?id=rJXTf9Bxg).
        # To preserve sum of sample weights for the auxiliary classifier,
        # we assign sample weight of 2 to the real images.
        disc_sample_weight = [np.ones(2 * batch_size),
                              np.concatenate((np.ones(batch_size) * 2,
                                              np.zeros(batch_size)))]
        # TODO: what is disc_sample_weight doing?
        
        epoch_gen_loss = []
        epoch_disc_loss = []

        
        for index in range(num_batches):
            
            # get a batch of real images
            image_batch = x_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]

            # generate a new batch of noise
            noise = np.random.uniform(-1, 1, (batch_size, latent_size))

            # sample some labels from p_c
            sampled_labels = np.random.randint(0, num_classes, batch_size)
            
            # generate a batch of fake images
            generated_images = generator.predict([noise, sampled_labels.reshape((batch_size, 1))])

            x = np.concatenate((image_batch, generated_images))
            
            # use one-sided soft real/fake labels
            # Salimans et al., 2016
            # https://arxiv.org/pdf/1606.03498.pdf (Section 3.4)
            soft_zero, soft_one = 0, 0.95
            y = np.array([soft_one] * batch_size + [soft_zero] * batch_size) # is real probability
            aux_y = np.concatenate((label_batch, sampled_labels), axis=0) # image class
            
            # train the discriminator
            disc_loss = discriminator.train_on_batch(x, [y, aux_y], sample_weight=disc_sample_weight)
            epoch_disc_loss.append(disc_loss)

            # generate another batch of noise. train the generator over an 
            # identical number of images as the discriminator.
            noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))
            sampled_labels = np.random.randint(0, num_classes, 2 * batch_size)

            # we want to train the generator to trick the discriminator
            # For the generator, we want all the {fake, not-fake} labels to say
            # not-fake
            trick = np.ones(2 * batch_size) * soft_one

            gen_loss = combined.train_on_batch(
                [noise, sampled_labels.reshape((-1, 1))], 
                [trick, sampled_labels])
            epoch_gen_loss.append(gen_loss)
          
        print('Testing for epoch {}:'.format(epoch))
        
        # evaluate the testing loss here

        # generate a new batch of noise and sample labels from p_c,
        # then generate new images from that
        noise = np.random.uniform(-1, 1, (num_test, latent_size))
        sampled_labels = np.random.randint(0, num_classes, num_test)
        generated_images = generator.predict([noise, sampled_labels.reshape((-1, 1))])

        x = np.concatenate((x_test, generated_images))
        y = np.array([1] * num_test + [0] * num_test)
        aux_y = np.concatenate((y_test, sampled_labels), axis=0)

        # evaluate the discriminator
        discriminator_test_loss = discriminator.evaluate(x, [y, aux_y]) 

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
        
        
        # make new noise
        noise = np.random.uniform(-1, 1, (2 * num_test, latent_size))
        sampled_labels = np.random.randint(0, num_classes, 2 * num_test)
        trick = np.ones(2 * num_test)

        generator_test_loss = combined.evaluate(
                [noise, sampled_labels.reshape((-1, 1))],
                [trick, sampled_labels])
        
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0) 
        
        # generate an epoch report on performance
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format('component', *discriminator.metrics_names))
        print('-' * 65)
        
        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.4f} | {3:<5.4f}'
        print(ROW_FMT.format('generator (train)', *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)', *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)', *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)', *test_history['discriminator'][-1]))

        generator.save_weights('generator_epoch_{0:03d}.hdf5'.format(epoch), True)
        discriminator.save_weights('discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)

        
    with open('keras_acgan_history.pkl', 'wb') as f:
        pickle.dump({'train': train_history, 'test': test_history}, f)    
