#dataloader by jimmy


import torch.utils.data as data

from os import listdir

import numpy as np
import random


def slit_train_val(input_path, target_path, phase):
	input_data = np.load(input_path)
	target_data = np.load(target_path)
	# input_data = transpose_data(input_data)
	# target_data = transpose_data(target_data)
	data_size = target_data.shape[0]
	print("total data size is " + str(data_size))

	# dataid = list(range(data_size))
	# #random.shuffle(dataid)
	# shuffle_id = np.load("./save_data/shuffle_id.npy")
	
	train_size = int(0.9*data_size)
	print("total training data size is " + str(train_size))
	train_id = input_data[:train_size]
	val_id = input_data[train_size:]
	#np.save("val_id.npy", val_id)
	if phase == "train":
		return input_data[train_id, :], target_data[train_id]
	elif phase == "test":
		return input_data[ val_id, :], target_data[val_id]
	else:
		print("Wrong phase information")
		return None

def transpose_data(data):
	return np.transpose(data, (2,0,1,3))

class DataLoader(data.Dataset):
	"""docstring for DataLoader"""
	def __init__(self, input_path, target_path, phase):
		super(DataLoader, self).__init__()
#		self.arg = arg
		self.input, self.target= slit_train_val(input_path, target_path, phase)
		
	
	def __getitem__(self, index):

		return self.input[index,:], self.target[index]

	def __len__(self):
		return self.input.shape[0]