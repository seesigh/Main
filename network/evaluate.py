import tensorflow as tf
import dataGen


#<======================_INPUT_DATA_======================>
data_dir = 'E:\\programming\\projects\\see_sign\\Main\\network\\dataset\\data_6'
test_dir = 'E:\\programming\\projects\\see_sign\\Main\\network\\dataset\\test_6_1'		# mid test (+night)
test_dir2 = 'E:\\programming\\projects\\see_sign\\Main\\network\\dataset\\test_6_2'	# heavy test (+night, big distance, angle)
test_dir3 = 'E:\\programming\\projects\\see_sign\\Main\\network\\dataset\\test_6_3'	# light test (+angle, mid distance)

ds = dataGen.DataSetGenerator(test_dir3)
data = ds.get_data_set(data_set_size=60, image_size=(128, 128), allchannel=True)
f1, l1 = data


#<======================_LOAD_FULL_MODEL_======================>
model = tf.keras.models.load_model('./full_model/model_project3_inits_rgb_6c_40ep.h5')
#model = tf.keras.models.load_model('./full_model/model_project3_inits_10ep.h5')

'''
#<======================_LOAD_CLEAR_MODEL_======================>
with open('./saved_model/model_project1.json', 'rt', encoding='utf-8') as fileobj:
	json_model = fileobj.read()
model = tf.keras.models.model_from_json(json_model)
model.compile(
	optimizer='adam',							#tf.train -> optimizers
	loss='categorical_crossentropy', 			#tf.keras.losses
	metrics=['accuracy'])			 			#tf.keras.metrics

#<======================_WEIGHTS_LOAD_======================>
model.load_weights('./weight/model_project1_1ep')
'''

#<======================_EVALUATE_MODEL_======================>
evaluate = model.evaluate(f1, l1, verbose=0)
print('lose: {}\naccuracy: {}'.format(evaluate[0], evaluate[1]))