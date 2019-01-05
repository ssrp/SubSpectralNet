# import the requirements
import tensorflow as tf
config = tf.ConfigProto()
# dynamic usage of GPU RAM
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
from keras import backend as K
from keras import layers, models, optimizers
K.set_image_data_format('channels_last')
import h5py
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.models import Model
from keras import regularizers
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate
from keras import callbacks
import math
import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Lambda
from keras.layers.normalization import BatchNormalization
import json
from numpy.random import seed
seed(1028)


# Load the dataset generated from DCASE Baseline Code
x_trainL = np.load('X_train_left.npy')
x_trainR = np.load('X_train_right.npy')

x_testL = np.load('X_validation_left.npy')
x_testR = np.load('X_validation_right.npy')

Y_TRAIN = np.load('Y_train.npy')
Y_TEST = np.load('Y_validation.npy')
x_train = np.concatenate([x_trainL, x_trainR], axis = 3)
x_test = np.concatenate([x_testL, x_testR], axis = 3)


######## Set the Parameters ##########
# Number of mel-bins of the Magnitude Spectrogram
melSize = x_train.shape[1]

# Sub-Spectrogram Size
splitSize = 20

# Mel-bins overlap
overlap = 10

# Time Indices
timeInd = 500

# Channels used
channels = 2



####### Generate the model ###########
inputLayer = Input((melSize,timeInd,channels))
subSize = splitSize/10
i = 0
inputs = []
outputs = []
toconcat = []
y_test = []
y_train = []
y_test.append(Y_TEST)
y_train.append(Y_TRAIN)
while(overlap*i <= melSize - splitSize):

	# Create Sub-Spectrogram
    INPUT = Lambda(lambda inputLayer: inputLayer[:,i*overlap:i*overlap+splitSize,:,:],output_shape=(splitSize,timeInd,channels))(inputLayer)

    # First conv-layer -- 32 kernels
    CONV = Conv2D(32, kernel_size=(7, 7), padding='same', kernel_initializer="he_normal")(INPUT)
    CONV = BatchNormalization(mode=0, axis=1,
                             gamma_regularizer=l2(0.0001),
                             beta_regularizer=l2(0.0001))(CONV)
    CONV = Activation('relu')(CONV)

    # Max pool by SubSpectrogram <mel-bin>/10 size. For example for sub-spec of 30x500, max pool by 3 vertically.
    CONV = MaxPooling2D((subSize,5))(CONV)
    CONV = Dropout(0.3)(CONV)

    # Second conv-layer -- 64 kernels
    CONV = Conv2D(64, kernel_size=(7, 7), padding='same',
                         kernel_initializer="he_normal")(CONV)
    CONV = BatchNormalization(mode=0, axis=1,
                             gamma_regularizer=l2(0.0001),
                             beta_regularizer=l2(0.0001))(CONV)
    CONV = Activation('relu')(CONV)

    # Max pool
    CONV = MaxPooling2D((4,100))(CONV)
    CONV = Dropout(0.30)(CONV)

    # Flatten
    FLATTEN = Flatten()(CONV)
    
    OUTLAYER = Dense(32, activation='relu')(FLATTEN)
    DROPOUT = Dropout(0.30)(OUTLAYER)
    
    # Sub-Classifier Layer
    FINALOUTPUT = Dense(10, activation='softmax')(DROPOUT)

    # to be used for model output
    outputs.append(FINALOUTPUT)

    # to be used for global classifier
    toconcat.append(OUTLAYER)

    y_test.append(Y_TEST)
    y_train.append(Y_TRAIN)

    i = i+1

x = Concatenate()(toconcat)

# Automatically chooses appropriate number of hidden layers -- in a factor of 2. 
# For example if  the number of sub-spectrograms is 9, we have 9*32 neurons by 
# concatenating. So number of hidden layers would be 5 -- 512, 256, 128, 64
numFCs = int(math.log(i*32, 2))
print(i*32)
print(numFCs)
print(math.pow(2, numFCs))
neurons = math.pow(2, numFCs)

# Last layer before Softmax is a 64-neuron hidden layer
while(neurons >= 64):
    x = Dense(int(neurons), activation='relu')(x)
    x = Dropout(0.30)(x)
    neurons = neurons / 2

# softmax -- GLOBAL CLASSIFIER
out = Dense(10, activation='softmax')(x)
outputs.append(out)

# Instantiate the model
classification_model = Model(inputLayer, outputs)

# Summary
print(classification_model.summary())

# Create the CSV logs.
type = 'SubSpectralNet-' + str(melSize) + '_' + str(splitSize) + '_' + str(overlap)
log = callbacks.CSVLogger('result/log_' + type + '.csv')
tb = callbacks.TensorBoard(log_dir='result/tensorboard-logs')
checkpoint = callbacks.ModelCheckpoint('result/model_' + type + '.h5', monitor='val_acc',verbose=1)

# Compile the model
classification_model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer=Adam(lr=0.001), 
              metrics=['accuracy']) # reporting the accuracy

# Train the model
classification_model.fit(x_train, y_train, batch_size=16, epochs=200,
 callbacks=[log,tb,checkpoint],verbose=1,validation_data=(x_test, y_test), shuffle=True)

# classification_model.load_weights('resultsFinal/model_200.30.10.88.h5')
# y_pred = classification_model.predict(x_test)
# np.save('y_pred.npy', y_pred);
