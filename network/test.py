import dataGen
import os
import IPython.display as display
import numpy as np

data_dir = 'E:\\programming\\projects\\see_sign\\Main\\network\\dataset\\data'
test_dir = 'E:\\programming\\projects\\see_sign\\Main\\network\\dataset\\test'

def rename_data(data_dir):
	ls = os.listdir(data_dir)
	for filename in ls:
		old_name = os.path.join(data_dir,filename)
		filename = filename.replace('_','.')
		new_name = os.path.join(data_dir,filename)
		os.rename(old_name, new_name)

rename_data(test_dir)
dataGen.separateData(test_dir)

'''
ds = dataGen.DataSetGenerator(data_dir)
label_names = ds.data_labels
data_paths = ds.data_info

data = ds.get_data_set(data_set_size=50, image_size=(96, 96), allchannel=False)
f, l = data
f = np.asarray(f)
l =  np.asarray(l)
print('\nf: {}, \nl: {}'.format(f.shape, l.shape))
print(l)
#print(data.shape)

'''
