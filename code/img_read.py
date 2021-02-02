import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import myparse
import os
import re
# tf.enable_eager_execution()

def natural_key(string_):
	return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass

def load_and_preprocess_image(path):
	img = cv2.imread(path)
	img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = img.astype(float)
	img /= 255.0
	return img

def disp(temp):
	# displays list of images in a tensor
	n_img, _,_,_ = temp.shape
	for i in range(n_img):
		r,c, ch = temp[i].shape
		cv2.imshow('img',temp[i])
		cv2.waitKey(0)
		# plt.imshow(np.reshape(temp[i], (r,c)))
		# plt.show()

def split_data(img, label, n_):

	train_ratio = int((3*n_)/4)
	train_img = img; train_lbl = label
	test_img = img[train_ratio:]; test_lbl = label[train_ratio:]
	return (train_img, train_lbl), (test_img, test_lbl)


def get_data():
	# /home/goutham/cv/jcb_track/Movies/data/full_arms_450/

	# img_check = os.path.isfile('./image_data.npy')
	# lbl_check = os.path.isfile('./label_data.npy')

	# if not(img_check and lbl_check):
		
	data_root = pathlib.Path('../train/')
	all_img_path = list(data_root.glob('./*.jpeg'))
	all_image_paths = [str(path) for path in all_img_path]
	all_image_paths = sorted(all_image_paths, key=natural_key)
	
	image_data = map(load_and_preprocess_image, all_image_paths)
	image_data = np.asarray(list(image_data))
	label_data = myparse.get_labels()
	
	# np.save('image_data.npy', image_data)
	# np.save('label_data.npy', label_data)
	# else:
		
		# image_data = np.load('image_data.npy')
		# label_data = np.load('label_data.npy')
	# print(image_data.shape)
	n_im,_,_ =  image_data.shape
	
	# trimmimg image data to multiple of 3 (since using 3d-cnn of 3 frames)
	image_data = image_data[0:n_im-1]
	

	# reshaping into 3d frames
	n_frames = 2
	n_ = int((n_im-1)/n_frames) #13

	image_data = np.reshape(image_data, (n_, n_frames, 225, 225, 1))


	n_ = label_data.shape[0]
	train, test = split_data(image_data, label_data, n_)
	
	# print(image_data.shape, label_data.shape, type(image_data), type(label_data))
	# print(train[0].shape, train[1].shape, test[0].shape, test[1].shape); 
	# print(train[1][30])
	# disp(train[0][30])


	train_data = tf.data.Dataset.from_tensor_slices(train)
	# train_data = train_data.batch(int((2*n_)/3)+2)
	train_data = train_data.batch(1)

	test_data = tf.data.Dataset.from_tensor_slices(test)
	# test_data = test_data.batch(n_ - int((2*n_)/3)-2)
	test_data = test_data.batch(1)

	return train_data, test_data



# get_data()



# print(test[0].shape, train[0].shape)
# print(test[1].shape, train[1].shape)
# print(type(image_data), type(labels))

# def preprocess_image(image):
# image = tf.read_file(path)
# 	# image = tf.image.decode_jpeg(image, channels=3)
# 	# image = tf.image.resize_images(image, im_shape)
# 	# arg = tf.convert_to_tensor(image, dtype=tf.float32)
# 	# image = tf.image.resize_images(blur, [28, 28])
# 	image /= 255.0  # normalize to [0,1] range
# 	return image

# res = cv2.GaussianBlur(img,(7,7),0)
# res = cv2.resize(img,(228, 228), interpolation = cv2.INTER_CUBIC)

# path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
# image_ds = path_ds.map(all_image_paths)
# for n,image in enumerate(image_ds.take(4)):


