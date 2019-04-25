import tensorflow as tf
from input_data import load_data, transform_to_dense, transform_to_conv
import dataGen


#<======================_INPUT_DATA_======================>
data_dir = 'E:\\programming\\projects\\see_sign\\Main\\network\\dataset\\data'
test_dir = 'E:\\programming\\projects\\see_sign\\Main\\network\\dataset\\test'

ds = dataGen.DataSetGenerator(test_dir)
data = ds.get_data_set(data_set_size=25, image_size=(96, 96), allchannel=True)
f1, l1 = data


#<======================_LOAD_FULL_MODEL_======================>
model = tf.keras.models.load_model('./full_model/model_project3_inits_rgb_40ep.h5')
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