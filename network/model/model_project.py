from tensorflow import keras as ks


#<======================_STATIC_DATA_======================>
img_size = 128
channels_num = 3
class_num = 6


#<======================_CREATE_MODEL_======================>
# послідовна модель, шар за шаром
model = ks.Sequential()
model.add(ks.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(img_size, img_size, channels_num), padding='same', kernel_initializer='he_normal'))
model.add(ks.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(ks.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))	# 64

model.add(ks.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(ks.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(ks.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))) # 32

model.add(ks.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(ks.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(ks.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(ks.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))) # 16

model.add(ks.layers.Flatten()) #128*16

model.add(ks.layers.Dense(512, activation='sigmoid', kernel_initializer='glorot_normal'))
model.add(ks.layers.Dropout(rate=0.5))
model.add(ks.layers.Dense(1096, activation='sigmoid', kernel_initializer='glorot_normal'))
model.add(ks.layers.Dropout(rate=0.5))
model.add(ks.layers.Dense(class_num, activation='softmax', kernel_initializer='he_normal'))


#<======================_SAVE_CLEAR_MODEL_======================>
json_model = model.to_json()
with open('../saved_model/model_project3_inits_rgb_128.json', 'wt', encoding='utf-8') as fileobj:
	fileobj.write(json_model)
