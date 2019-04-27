import tensorflow as tf
from input_data import load_data, transform_to_dense, transform_to_conv
import matplotlib.pyplot as plt
import numpy as np
import dataGen


#<======================_INPUT_DATA_======================>
data_dir = 'E:\\programming\\projects\\see_sign\\Main\\network\\dataset\\data_6'
test_dir = 'E:\\programming\\projects\\see_sign\\Main\\network\\dataset\\test_6_1'		# mid test (+night)
test_dir2 = 'E:\\programming\\projects\\see_sign\\Main\\network\\dataset\\test_6_2'	# heavy test (+night, big distance, angle)
test_dir3 = 'E:\\programming\\projects\\see_sign\\Main\\network\\dataset\\test_6_3'	# light test (+angle, mid distance)

ds = dataGen.DataSetGenerator(test_dir3)
data = ds.get_data_set(data_set_size=60, image_size=(128, 128), allchannel=True)
f1, l1 = data
first,last = 0,60


#<======================_LOAD_FULL_MODEL_======================>
model = tf.keras.models.load_model('./full_model/model_project3_inits_rgb_6c_40ep.h5')
#model.load_weights('./weight/model1')


#<======================_MODEL_PREDICT_======================>
predict = model.predict(f1[first:last], batch_size=1)

print(predict[:,0:-1].argmax(axis=1))
print(l1[first:last].argmax(axis=1))


# <==============================_SHOW_==============================>
f1 = np.reshape(f1, (-1,128,128,3))
for i in range(first,last):
	show_imgs = plt.imshow(f1[i], cmap='gray')
	#print(f1[i])
	plt.show()
