# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Machine Learning Tutorial 
ESMRMB Lecture, Berlin September 2018
Teodora Chitiboi, Sandy Engelhardt, Hans Meine 

This tutorial provides an introduction into CNN Autoencoders for reconstruction of MRI data.

Data set
"The 20 normal MR brain data sets and their manual
segmentations were provided by the Center for Morphometric Analysis at
Massachusetts General Hospital and are available at
http://www.cma.mgh.harvard.edu/ibsr/.
"""

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt


images = np.load('/home/engelhardt/data/ESMRMB/IBSR_v2_resampled_cropped_8bit_64x64.npz_FILES/input.npy')


#Divide dataset into train and test
x_train = images[:2500,:,:]
x_test  = images[2500:,:,:]
print(x_train.shape)
print(x_test.shape)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = np.reshape(x_train, (len(x_train), 64, 64, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 64, 64, 1))  # adapt this if using `channels_first` image data format


input_img = Input(shape=(64, 64, 1))  # adapt this if using `channels_first` image data format

# Convolutional layer Kernel 3x3
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
# Max Pooling Kernel 2x2
x = MaxPooling2D((2, 2), padding='same')(x)
# Convolutional layer Kernel 3x3
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# Max Pooling  Kernel 2x2
x = MaxPooling2D((2, 2), padding='same')(x)
# Convolutional layer Kernel 3x3
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# Max Pooling  Kernel 2x2
encoded = MaxPooling2D((2, 2), padding='same')(x)

print('Encoded dimensions should be (8,8,8):')
print(encoded.shape)


# Convolutional layer Kernel 3x3
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
# Upsampling Kernel 2x2
x = UpSampling2D((2, 2))(x)
# Convolutional layer Kernel 3x3
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# Upsampling Kernel 2x2
x = UpSampling2D((2, 2))(x)
# Convolutional layer Kernel 3x3
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
# Upsampling Kernel 2x2
x = UpSampling2D((2, 2))(x)
# Convolutional layer Kernel 3x3
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Set optimizer and loss
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# train autoencoder
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=16,
                shuffle=True,
                validation_data=(x_test, x_test))

# test the autoencoder
decoded_imgs = autoencoder.predict(x_test)

# image display 
n = 6 # number of images to display
plt.figure(figsize=(12, 4))
for i in range(n):
    
    # display image
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(64, 64))
    plt.gray()
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)

    # display reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(64, 64))
    plt.gray()
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
plt.show()