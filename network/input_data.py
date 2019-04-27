import tensorflow as tf
import numpy as np


def load_data():
	data = tf.keras.datasets.mnist.load_data()
	train,test = data
	f1,l1 = train
	f2,l2 = test
	f1 = f1/np.float32(255)+0.001
	f2 = f2/np.float32(255)+0.001
	l1 = tf.keras.utils.to_categorical(l1, 10)
	l2 = tf.keras.utils.to_categorical(l2, 10)

	train = f1, l1
	test = f2, l2
	data = train, test
	return data


def transform_to_conv(data, img_size, channels_num):
	train, test = data
	f1,l1 = train
	f2,l2 = test
	f1 = np.reshape(f1,(-1, img_size, img_size, channels_num))
	f2 = np.reshape(f2,(-1, img_size, img_size, channels_num))
	train = f1, l1
	test = f2, l2
	data = train,test
	return data


def transform_to_dense(data):
	train, test = data
	f1,l1 = train
	f2,l2 = test
	f1 = np.reshape(f1,(-1, 784))
	f2 = np.reshape(f2,(-1, 784))
	train = f1, l1
	test = f2, l2
	data = train,test
	return data


if __name__ == '__main__':
	print('\ninput_data:')
	data = load_data()
	train, test = transform_to_dense(data)
	f1,l1=train
	f2,l2=test
	print('train: {}\ntest: {}'.format(np.array(f1).shape, np.array(f2).shape))

