#Code taken from https://www.tensorflow.org/tutorials/images/classification

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.datasets import cifar10

def generate_model(dropout,HIDDEN_UNITS,activation,optimizer,x_train):
    if optimizer is 'sgd':
        optimizer = tensorflow.keras.optimizers.SGD(lr=0.1 ,decay=1e-6, momentum=0.9, nesterov=True)
    if optimizer is 'adadelta':
        optimizer = tensorflow.keras.optimizers.Adadelta(learning_rate=0.1, rho=0.95)
	
    if activation == "leaky":
		model_new = Sequential([
    	Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]),
	    # BatchNormalization(),
        MaxPooling2D(),
	    Dropout(dropout),
	    Conv2D(32, 3, padding='same', activation='relu'),
	    # BatchNormalization(),
        MaxPooling2D(),
	    Conv2D(64, 3, padding='same', activation='relu'),
	    # BatchNormalization(),
        MaxPooling2D(),
	    Dropout(dropout),
	    Flatten(),
	   	Dense(HIDDEN_UNITS),
	   	LeakyReLU(alpha=0.3),
        # BatchNormalization(),
	   	Dense(10, activation='sigmoid')
		])
    else:
		model_new = Sequential([
	    Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]),
        # BatchNormalization(),
	    MaxPooling2D(),
	    Dropout(dropout),
	    Conv2D(32, 3, padding='same', activation='relu'),
	    # BatchNormalization(),
        MaxPooling2D(),
	    Conv2D(64, 3, padding='same', activation='relu'),
	    # BatchNormalization(),
        MaxPooling2D(),
	    Dropout(dropout),
	    Flatten(),
	   	Dense(HIDDEN_UNITS,activation =activation),
        # BatchNormalization(),
	   	Dense(10, activation='sigmoid')	])

    print(optimizer)
    model_new.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    model_new.summary()
    return model_new



batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# activation =['leaky','relu','tanh','sigmoid','softsign']
activation = ['relu']
optimizers = [ 'adam','sgd','rmsprop','adagrad','adadelta']
test_accuracies = []
accuracies = []
val_accuracies = []
losses = []
val_losses =[]
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    shear_range=0.,  # set range for random shear
    zoom_range=0.,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.33)


datagen.fit(x_train)


test_accuracies = []
accuracies = []
val_accuracies = []
losses = []
val_losses =[]

for idx in range (0,len(optimizers)):
	name = 'optimizers/cnn ' +str(optimizers[idx] +'.h5')
	model = generate_model(0.2,512,'leaky',optimizers[idx],x_train)

	checkpoint = ModelCheckpoint(name, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
	early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
	

	history = model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size,subset ='training'),
                      				 epochs=epochs,
                      				 validation_data= datagen.flow(x_train, y_train,
                                     batch_size=batch_size,subset ='validation'),
                        			callbacks=[checkpoint,early])
	model.load_weights(name)
	score = model.evaluate(x_test, y_test, verbose=0)

	accuracies.append(history.history['acc'])
	val_accuracies.append(history.history['val_acc'])

	losses.append(history.history['loss'])
	val_losses.append(history.history['val_loss'])
	test_accuracies.append(score)

epochs_range = range(epochs)

accuracies_df = pd.DataFrame([array for array in accuracies] )
accuracies_df.to_csv("optimizers/accuracies.csv")

val_accuracies_df = pd.DataFrame([array for array in val_accuracies] )
val_accuracies_df.to_csv("optimizers/val_accuracies.csv")

test_accuracies_df = pd.DataFrame(test_accuracies )
test_accuracies_df.to_csv("optimizers/test_accuracies.csv")

losses_df = pd.DataFrame([array for array in losses] )
losses_df.to_csv("optimizers/losses.csv")

val_losses_df = pd.DataFrame([array for array in val_losses] ) 
val_losses_df.to_csv("optimizers/val_losses.csv")
