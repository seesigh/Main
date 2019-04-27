import cv2
from os.path import isfile, join
import numpy as np
import os
from random import shuffle
from shutil import copyfile
import pickle


# розділ зображень у різні директорії відповідно до назви(cat, dog...)
# назви котрих представлені як " imgClass.imgNum.jpg "
# директорії будуть збережені в data_dir
def separateData(data_dir):
	# listdir  -   список файлів та директорій: ['one.jpg', 'test.py', dir2]
	# join  -  з'єднує адреси, опрацьовує слеші між ними(є чи нема)
	# copyfile  -  копіює вміст file1 у file2 , якщо file2 нема то створює
	# makedirs  -  створює директорію
	for file_name in os.listdir(data_dir):
		if isfile(join(data_dir, file_name)):
			tokens = file_name.split('.')
			if tokens[-1] == 'jpg':
				image_path = join(data_dir, file_name)
				if not os.path.exists(join(data_dir, tokens[0])):
					os.makedirs(join(data_dir, tokens[0]))
				copyfile(image_path, join(join(data_dir, tokens[0]), file_name))
				os.remove(image_path)


# для обробки зображень 
# перетворення їх в робочий датасет
class DataSetGenerator:
    def __init__(self, data_dir):
        # адреса директорії з директоріями-класами зображень
        self.data_dir = data_dir
        # масив адрес директорій-класів зображень
        self.data_labels = self.get_data_labels()
        # вектор масивів із адресами зображень відповідного класу-масиву
        self.data_info = self.get_data_paths()


    # добавляє назви директорій з data_dir в масив та повертає його
    # назви директорій ([cat, dog] - класів зображень)
    def get_data_labels(self):
        data_labels = []
        for filename in os.listdir(self.data_dir):
            if not isfile(join(self.data_dir, filename)):
                data_labels.append(filename)
        return data_labels


    # повертає вектор перемішаних масивів із адресами зображень відповідного класу
    # data_paths  -  [[cat1,cat2], [dog2,dog1]]
    def get_data_paths(self):
        data_paths = []
        # проходить директорії [cat, dog]
        for label in self.data_labels:
            img_lists=[]
            path = join(self.data_dir, label)
            # проходить зображення в директоріях [cat1, cat2]
            for filename in os.listdir(path):
                tokens = filename.split('.')
                if tokens[-1] == 'jpg':
                    image_path=join(path, filename)
                    img_lists.append(image_path)
            shuffle(img_lists)
            data_paths.append(img_lists)
        return data_paths


    # генератор, бере з кожної директорії-класу по зображенню batch_size разів
    # і робить yield (inputs, targets) 
    # поки відправить максимальну кк зображень з кожного класу(counter), кратну batch_size
    # min_batch_size = кк класів зображень - len(self.data_labels)
    def get_mini_batches(self, batch_size=10, image_size=(200, 200), allchannel=True):
       images = []
       labels = []
       empty=False
        # кк взятих зображень з кожного класу де вони ше є
       counter=0
       each_batch_size=int(batch_size/len(self.data_info))
       if each_batch_size == 0:
          print ('\ntoo small batch, minimal_batch_size = {} (num of img classes)\n'.format(len(self.data_labels)))
          raise Exception

       while True:
       	# проходить по директоріях-класах зображень
       	# бере по 1 зображенні для кожного класу
       	# якщо в найменшому класі закінчились то бере з більших
           for i in range(len(self.data_labels)):
           	# вектор нулів
               label = np.zeros(len(self.data_labels),dtype=int)
               # 1 на індексі правильного класу
               label[i] = 1
               # чи кк зображень у директорії == кк взятих
               if len(self.data_info[i]) < counter+1:
                   empty=True
                   # бо в іншій директорій може бути більше зображень
                   continue
               empty=False
               img = cv2.imread(self.data_info[i][counter])
               img = self.resizeAndPad(img, image_size)
               # перетворює в grayscale 
               if not allchannel:
                   img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                   img = np.reshape(img, (img.shape[0], img.shape[1], 1))
              
               images.append(img)
               labels.append(label)
           counter+=1
           # якщо взято всі зображення
           if empty:
               break
           # якщо кількість взятих зображень із кожного класу кратна batch_size/кк класів
           # повертає кортеж (input, target)
           if (counter)%each_batch_size == 0:
               yield np.array(images,dtype=np.float32)/256+0.001, np.array(labels,dtype=np.float32)
               del images
               del labels
               images=[]
               labels=[]


    def get_data_set(self, data_set_size=50, image_size=(200, 200), allchannel=True):
      data_set = list(self.get_mini_batches(batch_size=data_set_size, image_size=image_size, allchannel=allchannel))[0]
      return data_set


    # збереження назв директорій-класів зображень
    def save_labels(self, path):
        pickle.dump(self.data_labels, open(path,"wb"))


    # масштабування зображення до вказаних розмірів
    # спочатку до квадратної форми
    # тоді до вказаних розмірів 
    def resizeAndPad(self, img, size):
        h, w = img.shape[:2]

        sh, sw = size
        # interpolation method
        if h > sh or w > sw:  		# вкорочення
            interp = cv2.INTER_AREA
        else: 						# розширення
            interp = cv2.INTER_CUBIC

        # відношення ширини до висоти (px)
        aspect = w/h

        # width > height
        # зображення -> [w,w]
        # лишні місця заповнюються 0 (по половині зверху і знизу)
        if aspect > 1:
            new_shape = list(img.shape)
            new_shape[0] = w
            new_shape[1] = w
            new_shape = tuple(new_shape)
            new_img=np.zeros(new_shape, dtype=np.float32)
            h_offset=int((w-h)/2)
            new_img[h_offset:h_offset+h, :, :] = img.copy()

        # width < height
        # зображення -> [h,h]
        # лишні місця заповнюються 0 (по половині зліва і справа)
        elif aspect < 1:
            new_shape = list(img.shape)
            new_shape[0] = h
            new_shape[1] = h
            new_shape = tuple(new_shape)
            new_img = np.zeros(new_shape,dtype=np.float32)
            w_offset = int((h-w) / 2)
            new_img[:, w_offset:w_offset + w, :] = img.copy()
        
        # width = height
        # нічо не міняється
        else:
            new_img = img.copy()
        
        # масштабування до потрібних розмірів
        # з використанням певного алгоритму
        # для вкорочення краще INTER_AREA, а розширення - INTER_CUBIC
        scaled_img = cv2.resize(new_img, size, interpolation=interp)

        return scaled_img


if __name__ == '__main__':
  separateData('./data/dogs-vs-cats/train')