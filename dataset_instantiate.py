import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class InstantiateDataset(Dataset):
	def __init__(self, img_path, class_name):
		self.imgs_path = str(img_path)
		self.data = []
		self.data.append([img_path, class_name])
		print(self.data)
		self.class_map = {"airplane": 0, "bird": 1, "car": 2, "cat": 3, "deer": 4, "dog": 5, "horse": 6, "monkey": 7, "ship": 8, "truck": 9}
		self.img_dim = (96, 96)
	
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_path, class_name = self.data[idx]
		img = Image.open(img_path).convert("RGB")
		img = img.resize(self.img_dim)
		class_id = self.class_map[class_name]
		img_tensor = transforms.ToTensor()(img)
		class_id = torch.tensor([class_id])
		return img_tensor, class_id