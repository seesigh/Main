import dataGen
import os
import IPython.display as display
import numpy as np

data_dir = 'E:\\programming\\projects\\see_sign\\Main\\network\\dataset\\data'

def rename_data(data_dir):
	ls = os.listdir(data_dir)
	for filename in ls:
		old_name = os.path.join(data_dir,filename)
		filename = filename.replace('_','.')
		new_name = os.path.join(data_dir,filename)
		os.rename(old_name, new_name)

ds = dataGen.DataSetGenerator(data_dir)
label_names = ds.data_labels
data_paths = ds.data_info

data = ds.get_mini_batches(batch_size=50, image_size=(96, 96), allchannel=False)
f, l = list(data)[0]
f = np.asarray(f)
l =  np.asarray(l)
print(f.shape)

#print(data.shape)


