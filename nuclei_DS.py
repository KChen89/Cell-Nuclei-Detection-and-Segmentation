'''
detect and segement potential nuclei in miscropic images (H&E stained)
@author: Kemeng Chen 
'''
import os
import numpy as np 
import cv2
from time import time
from util import*

def process(data_folder, model_name, format):
	patch_size=128
	stride=16
	file_path=os.path.join(os.getcwd(), data_folder)
	name_list=os.listdir(file_path)
	print(str(len(name_list)), ' files detected')
	model_path=os.path.join(os.getcwd(), 'models')
	model=restored_model(model_path, os.path.join(model_path, model_name))
	print('Start time:')
	print_ctime()

	for index, temp_name in enumerate(name_list):
		ts=time()
		print('process: ', str(index), ' name: ', temp_name)
		temp_path=os.path.join(file_path, temp_name)
		if not os.path.isdir(temp_path):
			continue
		result_path=os.path.join(temp_path, 'mask.png')
		temp_image=cv2.imread(os.path.join(temp_path, temp_name+format))
		if temp_image is None:
			raise AssertionError(temp_path, ' not found')
		batch_group, shape=preprocess(temp_image, patch_size, stride, temp_path)
		mask_list=sess_interference(model, batch_group)
		c_mask=patch2image(mask_list, patch_size, stride, shape)
		c_mask=cv2.medianBlur((255*c_mask).astype(np.uint8), 3)
		c_mask=c_mask.astype(np.float)/255
		thr=0.5
		c_mask[c_mask<thr]=0
		c_mask[c_mask>-thr]=255
		te=time()
		print('Time cost: ', str(te-ts))
	model.close_sess()
	print('Task done:')
	print_ctime()

def main():
	data_folder='data'
	model_name='nucles_model_v3'

if __name__ == '__main__':
	main()
