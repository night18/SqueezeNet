'''
=======================================================================================
Author: Chun-Wei Chiang
Date: 2019.02.24
Description: Train SqueezeNet network
=======================================================================================
'''
import tensorflow as tf
import model
import util
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical

cifar_10_dir = "cifar-10"
validation_number = 10000
train_number = 50000 - validation_number
epochs = 200

def testModel(model, x_test, y_test, learning_rate):
	score = model.evaluate(x_test, y_test)
	print("=============================================")
	print("Test perforemance of learning rate " + str(learning_rate))
	print('Test loss:'+ str(score[0]))
	print('Test accuracy:'+ str(score[1]))


def plot_performance(histories, name_list, isloss = True, isVal = False, isBoth = False):
	#isloss means whether plot loss. If True, plot loss, nor plot accuracy

	perforemance = 'loss' if isloss else 'acc'

	# print(perforemance)
	fig = plt.figure()

	for hist in histories:
		if isBoth:
			plt.plot(hist[perforemance])
			plt.plot(hist['val_' + perforemance])
		else:
			val = 'val_' if isVal else ''
			perforemance = val + perforemance
			plt.plot(hist[perforemance])

	plt.xticks(np.arange(0, epochs +1 , epochs/5 ))
	plt.ylabel(perforemance)
	plt.xlabel( "epochs" )
	plt.legend( name_list , loc=0)
	# plt.show()
	fig.savefig(perforemance + '.png')


if __name__ == '__main__':
	train_data, _, train_labels, test_data, _, test_labels, label_names = util.loadCIFAR10(cifar_10_dir)

	train_data, validation_data = train_data[ 0:train_number ], train_data[ train_number:50000 ]
	train_labels, validation_labels = train_labels[ 0:train_number ], train_labels[ train_number:50000 ]
	train_data, validation_data, test_data = train_data/255.0, validation_data/255.0 , test_data/255.0
	
	train_labels= to_categorical(train_labels,num_classes=10)
	validation_labels = to_categorical(validation_labels, num_classes=10)
	test_labels= to_categorical(test_labels,num_classes=10)

	learning_rate_list = [0.01, 0.001, 0.0001]
	histories = []
	name_list = []

	for x in learning_rate_list:
		squeezenet = model.loadModel(learning_rate = x)
		if squeezenet == None: 
			squeezenet = model.trainModel(train_data, train_labels, validation_data, validation_labels, epochs=epochs, learning_rate=x)
		testModel(squeezenet, test_data, test_labels, x)
		path = "history/squeezeNet_{}".format(x)
		histories.append( util.unpickle(path) )
		name = "{}".format(x)
		name_list.append( name )
		name_list.append( name + "_val" )

	plot_performance(histories, name_list, isloss = False, isBoth = True)
