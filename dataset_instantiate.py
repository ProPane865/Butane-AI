import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class InstantiateDataset(Dataset):
	def __init__(self, img_path, class_name):
		self.imgs_path = str(img_path)
		self.data = []
		self.data.append([img_path, class_name])
		print(self.data)
		self.class_map = {"plane": 0, "car": 1, "bird": 2, "cat": 3, "deer": 4, "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9}
		self.img_dim = (32, 32)
	
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_path, class_name = self.data[idx]
		img = cv2.imread(img_path)
		img = cv2.resize(img, self.img_dim)
		class_id = self.class_map[class_name]
		img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
		img_tensor = img_tensor.permute(2, 0, 1)
		class_id = torch.tensor([class_id])
		return img_tensor, class_id