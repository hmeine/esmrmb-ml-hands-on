# -*- coding: utf-8 -*-
"""
Machine Learning Tutorial 
ESMRMB Lecture, Berlin September 2018
Teodora Chitiboi, Sandy Engelhardt, Hans Meine 

This tutorial provides an introduction into Dense Autoencoders for reconstruction of MRI data.

Data set
"The 20 normal MR brain data sets and their manual
segmentations were provided by the Center for Morphometric Analysis at
Massachusetts General Hospital and are available at
http://www.cma.mgh.harvard.edu/ibsr/.
"""

from keras import layers
from keras.models import Model
import numpy as np
# use Matplotlib for displaying of results
import matplotlib.pyplot as plt

#%% Loading dataset 
#%% Use Github here
images = np.load('/home/engelhardt/data/ESMRMB/IBSR_v2_resampled_cropped_8bit_64x64.npz_FILES/input.npy')
print('Dimensions after loading:')
print(images.shape)

#%% Reduce dimensions to (NumberOfDatasets, Height, Width)
dataset = np.squeeze(images, axis=1);
print('Dimensions after dimensionality reduction:')
print(dataset.shape) # should be (2716,1,64,64)

#%% Dividing dataset into train and test set
x_train = dataset[:2500,:,:]
x_test  = dataset[2500:,:,:]

#%% Normalizing images to range [0...1]
X_Train = x_train.astype('float32') / 255.
X_Test = x_test.astype('float32') / 255.

#%% Flattening images into 64*64 = 4096 vector
X_Train = X_Train.reshape((len(X_Train), np.prod(X_Train.shape[1:])))
X_Test = X_Test.reshape((len(X_Test), np.prod(X_Test.shape[1:])))
print('Flattened dimensions of training data')
print(X_Train.shape)
print('Flattened dimensions of testing data')
print(X_Test.shape)

#%%  
# Create the network model
# mapping input vector with factor 128, (32*128 = 4096)
encode_dimension = 128

# Input placeholder
input_img = layers.Input(shape=(4096,))

# Encoding of the input
encoded_1 = layers.Dense(encode_dimension, activation='relu')(input_img)
encoded_2 = layers.Dense(encode_dimension, activation='relu')(encoded_1)

# Decoding/reconstruction of the input
decoded_1   = layers.Dense(4096, activation='sigmoid')(encoded_2)
decoded_2   = layers.Dense(4096, activation='sigmoid')(decoded_1)

# This maps the input to its reconstruction
# The aim is to fully recover the input image
Encoder_Decoder = Model(input_img, decoded_2)

# Set the optimizer (Adam is a popular choice), and the loss function
Encoder_Decoder.compile(optimizer = 'adam', loss='binary_crossentropy')
# further options: loss='mean_squared_error', 'mean_absolute_error'
# optimizer='sgd'

#%%  Train the autoencoder

#%% Potentially change num_epochs or batch_size
num_epochs = 25
Encoder_Decoder.fit(X_Train, X_Train,
                epochs=num_epochs,
                batch_size=16,
                shuffle=True,
                validation_data=(X_Test, X_Test))

#%%  Test the autoencoder using the model to predict unseen data
decoded_imgs = Encoder_Decoder.predict(X_Test)

#%% Following code is for displaying of results

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