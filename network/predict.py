import tensorflow as tf
from input_data import load_data, transform_to_dense, transform_to_conv
import matplotlib.pyplot as plt
import numpy as np
import dataGen


#<======================_INPUT_DATA_======================>
data_dir = 'E:\\programming\\projects\\see_sign\\Main\\network\\dataset\\data'
test_dir = 'E:\\programming\\projects\\see_sign\\Main\\network\\dataset\\test'

ds = dataGen.DataSetGenerator(test_dir)
data = ds.get_data_set(data_set_size=25, image_size=(96, 96), allchannel=False)
f1, l1 = data
first,last = 0,25


#<======================_LOAD_FULL_MODEL_======================>
model = tf.keras.models.load_model('./full_model/model_project2_100ep.h5')
#model.load_weights('./weight/model1')


#<======================_MODEL_PREDICT_======================>
predict = model.predict(f1[first:last], batch_size=1)
print(predict.argmax(axis=1))
print(l1[first:last].argmax(axis=1))

f1 = np.reshape(f1, (-1,96,96))


# <==============================_SHOW_==============================>
for i in range(first,last):
	show_imgs = plt.imshow(f1[i], cmap='gray')
	#print(f1[i])
	plt.show()