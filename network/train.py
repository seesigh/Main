import tensorflow as tf
import dataGen

# поміняти сигмоїди на релу
# try batch norm
# написати свою точність (без 6 класу)

#<======================_LOAD_INPUT_DATA_======================>
data_dir = 'E:\\programming\\projects\\see_sign\\Main\\network\\dataset\\data_6'
test_dir = 'E:\\programming\\projects\\see_sign\\Main\\network\\dataset\\test'
test_dir2 = 'E:\\programming\\projects\\see_sign\\Main\\network\\dataset\\test2'
test_dir3 = 'E:\\programming\\projects\\see_sign\\Main\\network\\dataset\\test3'

ds = dataGen.DataSetGenerator(data_dir)
data = ds.get_data_set(data_set_size=180, image_size=(128, 128), allchannel=True)
f1,l1 = data

#<======================_SET_CALLBACKS_======================>
# tensorboard --logdir ./log_dir
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./log_dir/model_project3_inits_rgb_c6_3', write_graph=True)
#stopCallBack = tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss')
callbacks = [tbCallBack]

'''
#<======================_LOAD_FULL_MODEL_======================>
model = tf.keras.models.load_model('./full_model/model_project3_inits_40ep.h5')
'''


#<======================_LOAD_CLEAR_MODEL_======================>
with open('./saved_model/model_project3_inits_rgb_new.json', 'rt', encoding='utf-8') as fileobj:
	json_model = fileobj.read()
model = tf.keras.models.model_from_json(json_model)
model.compile(
	optimizer='adam',							#tf.train -> optimizers
	loss='categorical_crossentropy', 			#tf.keras.losses
	metrics=['accuracy'])			 			#tf.keras.metrics


#<======================_TRAIN_MODEL_======================>
model.fit(
	f1,
	l1,
	batch_size=90,
	epochs=40,
	verbose=2,
	callbacks=callbacks
	)


#<======================_SAVE_WEIGHTS_MODEL_======================>
model.save('full_model/model_project3_inits_rgb_6c_40ep.h5')
model.save_weights('weight/model_project3_inits_rgb_6c_40ep')




