'''
=======================================================================================
Author: Chun-Wei Chiang
Date: 2019.02.24
Description: Train SqueezeNet network
=======================================================================================
'''

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Activation, concatenate, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle

sqz = 'sqz1'
exp = 'exp'
exp1 = 'exp1'
exp3 = 'exp3'
relu = 'relu_'
models_dir = "models"
history_dir = "history"
checkpoint_dir = "checkpoint"


def fire_module(prv_lyr, fire_id, squeeze = 3, expand = 4):
	s_id = 'fire' + str(fire_id) + '/'

	#squeeze layer
	sqz_layer = Conv2D( squeeze, kernel_size=(1,1), padding='same', name=s_id+sqz )(prv_lyr)
	sqz_layer = Activation( 'relu', name=s_id+relu+sqz )(sqz_layer)

	#expand layer
	#1*1
	exp1_layer = Conv2D( expand, kernel_size=(1,1), padding='same', name=s_id+exp1)(sqz_layer)
	exp1_layer = Activation( 'relu', name=s_id+relu+exp1)(exp1_layer)
	#3*3
	exp3_layer = Conv2D( expand, kernel_size=(3,3), padding='same', name=s_id+exp3)(sqz_layer)
	exp3_layer = Activation( 'relu', name=s_id+relu+exp3)(exp3_layer)

	cnct_layer = concatenate([exp1_layer, exp3_layer])

	return cnct_layer

def squeezeNet(input_img):
	inputs = Input(shape=(32,32,3))

	x = Conv2D(96, kernel_size=(4,4), padding='same', name='conv1' )(inputs)
	x = Activation('relu', name='relu_conv1')(x)
	x = MaxPool2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)

	x = fire_module(x, fire_id=2, squeeze=16, expand=64)
	x = fire_module(x, fire_id=3, squeeze=16, expand=64)
	x = fire_module(x, fire_id=4, squeeze=32, expand=128)		
	x = MaxPool2D(pool_size=(3,3), strides=(2,2), name='pool2')(x)

	x = fire_module(x, fire_id=5, squeeze=32, expand=128)
	x = fire_module(x, fire_id=6, squeeze=48, expand=192)
	x = fire_module(x, fire_id=7, squeeze=48, expand=192)
	x = fire_module(x, fire_id=8, squeeze=64, expand=256)
	x = MaxPool2D(pool_size=(3,3), strides=(2,2), name='pool3')(x)

	x = fire_module(x, fire_id=9, squeeze=64, expand=256)
	x = Conv2D(1000, kernel_size=(4,4), padding='same', name='conv10')(x)
	x = Activation('relu', name='relu_conv10')(x)

	x = Flatten()(x)
	x = Dense(10)(x)
	x = Activation('softmax', name='softmax')(x)

	model = Model(inputs, x, name='squeezeNet')

	return model

def trainModel(train_data, train_label, validation_data, validation_label, epochs=200, learning_rate=0.001 ):
	model = None
	h5_storage_path = models_dir + "/" + "squeezeNet_" + str(learning_rate) + ".h5"
	hist_storage_path = history_dir + "/" + "squeezeNet_" + str(learning_rate)
	checkpoint_path = checkpoint_dir + "/" + "squeezeNet_" + str(learning_rate) + ".hdf5"

	model = squeezeNet(train_data)
	model.compile(loss = tf.keras.losses.categorical_crossentropy,
						optimizer = SGD(lr = learning_rate),
						metrics = ['accuracy'])

	checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]

	#Fit the model
	hist = model.fit(
		train_data,
		train_label,
		epochs = epochs,
		batch_size = 64,
		validation_data=(validation_data, validation_label),
		callbacks=callbacks_list,
		verbose= 1)

	#save the model
	save_model(
		model,
		h5_storage_path,
		overwrite=True,
		include_optimizer=True
	)

	#Save the history of training
	with open(hist_storage_path, 'wb') as file_hist:
		pickle.dump(hist.history, file_hist)

	print("Successfully save the model at " + h5_storage_path)

	return model
