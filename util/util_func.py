'''
utility functions assisting nuclei detection and segmentation
@author: Kemeng Chen
'''
import numpy as np 
import cv2
import os
import sys
from time import time, ctime
from restored_model import restored_model

def print_ctime():
	current_time=ctime(int(time()))
	print(str(current_time))

def batch2list(batch):
	mask_list=list()
	for index in range(batch.shape[0]):
		mask_list.append(batch[index,:,:])
	return mask_list

def patch2image(patch_list, patch_size, stride, shape):	
	if shape[0]<patch_size:
		L=0
	else:
		L=math.ceil((shape[0]-patch_size)/stride)
	if shape[1]<patch_size:
		W=0
	else:
		W=math.ceil((shape[1]-patch_size)/stride)	

	full_image=np.zeros([L*stride+patch_size, W*stride+patch_size])
	bk=np.zeros([L*stride+patch_size, W*stride+patch_size])
	cnt=0
	for i in range(L+1):
		for j in range(W+1):
			full_image[i*stride:i*stride+patch_size, j*stride:j*stride+patch_size]+=patch_list[cnt]
			bk[i*stride:i*stride+patch_size, j*stride:j*stride+patch_size]+=np.ones([patch_size, patch_size])
			cnt+=1   
	full_image/=bk
	image=full_image[0:shape[0], 0:shape[1]]
	return image

def image2patch(in_image, patch_size, stride, blur=False, f_size=9):
	if blur is True:
		in_image=cv2.medianBlur(in_image, f_size)
		# in_image=denoise_bilateral(in_image.astype(np.float), 19, 11, 9, multichannel=False)
	shape=in_image.shape
	if shape[0]<patch_size:
		L=0
	else:
		L=math.ceil((shape[0]-patch_size)/stride)
	if shape[1]<patch_size:
		W=0
	else:
		W=math.ceil((shape[1]-patch_size)/stride)	
	patch_list=list()
	
	if len(shape)>2:
		full_image=np.pad(in_image, ((0, patch_size+stride*L-shape[0]), (0, patch_size+stride*W-shape[1]), (0,0)), mode='symmetric')
	else:
		full_image=np.pad(in_image, ((0, patch_size+stride*L-shape[0]), (0, patch_size+stride*W-shape[1])), mode='symmetric')
	for i in range(L+1):
		for j in range(W+1):
			if len(shape)>2:
				patch_list.append(full_image[i*stride:i*stride+patch_size, j*stride:j*stride+patch_size, :])
			else:
				patch_list.append(full_image[i*stride:i*stride+patch_size, j*stride:j*stride+patch_size])
	if len(patch_list)!=(L+1)*(W+1):
		raise ValueError('Patch_list: ', str(len(patch_list), ' L: ', str(L), ' W: ', str(W)))
	
	return patch_list

def list2batch(patches):
	'''
	covert patch to flat batch
	args:
		patches: list
	return:
		batch: numpy array
	'''
	patch_shape=list(patches[0].shape)

	batch_size=len(patches)
	
	if len(patch_shape)>2:
		batch=np.zeros([batch_size]+patch_shape)
		for index, temp in enumerate(patches):
			batch[index,:,:,:]=temp
	else:
		batch=np.zeros([batch_size]+patch_shape+[1])
		for index, temp in enumerate(patches):
			batch[index,:,:,:]=np.expand_dims(temp, axis=-1)
	return batch

def preprocess(input_image, patch_size, stride, file_path):
	f_size=5
	g_size=10
	shape=input_image.shape
	patch_list=image2patch(input_image.astype(np.float32)/255.0, patch_size, stride)
	num_group=math.ceil(len(patch_list)/g_size)
	batch_group=list()
	for i in range(num_group):
		temp_batch=list2batch(patch_list[i*g_size:(i+1)*g_size])
		batch_group.append(temp_batch)
	return batch_group, shape

def sess_interference(sess, batch_group):
	patch_list=list()
	for temp_batch in batch_group:
		mask_batch=sess.run_sess(temp_batch)[0]
		mask_batch=np.squeeze(mask_batch, axis=-1)
		mask_list=batch2list(mask_batch)
		patch_list+=mask_list
	return patch_list

def draw_edge(data_folder, format):
	folder_path=os.path.join(os.getcwd(), data_folder)
	name_list=os.listdir(folder_path)
	for index, _ in enumerate(name_list):
		print(name_list[index])
		sample_path=os.path.join(folder_path, name_list[index], name_list[index]+format)
		mask_path=os.path.join(folder_path, name_list[index], 'c_mask.png')
		sample=cv2.imread(sample_path)
		mask=cv2.imread(mask_path)
		if sample is None:
			raise AssertionError(sample_path, ' not exists')
		if mask is None:
			raise AssertionError(mask_path, ' not exists')
		edge=cv2.Canny(mask, 2, 5)
		check_image=np.copy(sample)
		check_image[:,:,1]=np.maximum(check_image[:,:,1], edge)
		cv2.imwrite(os.path.join(os.getcwd(), 'label', name_list[index]+'_label.png'), check_image.astype(np.uint8))
