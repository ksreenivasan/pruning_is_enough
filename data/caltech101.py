
import joblib
import cv2
import os
import time
import random
import numpy as np

from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load torch...!!!
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Load torchvision ...!!!
from torchvision import transforms


from args_helper import parser_args


class Caltech101:
	def __init__(self, args):
		super(Caltech101, self).__init__()


		image_paths = list(paths.list_images('./101_ObjectCategories'))

		data = []
		labels = []
		for img_path in image_paths:
				label = img_path.split(os.path.sep)[-2]
				if label == "BACKGROUND_Google":
						continue
				img = cv2.imread(img_path)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				
				data.append(img)
				labels.append(label)
				
		data = np.array(data)
		labels = np.array(labels)


		lb = LabelEncoder()
		labels = lb.fit_transform(labels)
		print(f"Total Number of Classes: {len(lb.classes_)}")

		train_transforms = transforms.Compose([
		    transforms.ToPILImage(),
		    transforms.Resize((224, 224)),
		    transforms.ToTensor(),
		    transforms.Normalize(mean = [0.485,0.456,0.406], std=[0.229,0.224,0.225]),
		])

		val_transform = transforms.Compose([
		    transforms.ToPILImage(),
		    transforms.Resize((224, 224)),
		    transforms.ToTensor(),
		    transforms.Normalize(mean = [0.485,0.456,0.406], std=[0.229,0.224,0.225]),
		])

		# divide the data into train, validation, and test set
		(X, x_val , Y, y_val) = train_test_split(data, labels, test_size=0.2,  stratify=labels,random_state=42)
		(x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.25, random_state=42)
		print(f"x_train examples: {x_train.shape}\nx_test examples: {x_test.shape}\nx_val examples: {x_val.shape}")


		train_data = CustomDataset(x_train, y_train, train_transforms)
		val_data = CustomDataset(x_val, y_val, val_transform)
		test_data = CustomDataset(x_test, y_test, val_transform)       

		self.train_loader = DataLoader(train_data, batch_size=BS, shuffle=True, num_workers=4)
		self.test_loader = DataLoader(val_data, batch_size=BS, shuffle=True, num_workers=4)
		self.num_classes = len(lb.classes_)

		#valLoader = DataLoader(val_data, batch_size=BS, shuffle=True, num_workers=4)
		#testLoader = DataLoader(test_data, batch_size=BS, shuffle=True, num_workers=4) 


# custom dataset class
class CustomDataset(Dataset):
    def __init__(self, images, labels= None, transforms = None):
        self.labels = labels
        self.images = images
        self.transforms = transforms
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        data = self.images[index][:]
        
        if self.transforms:
            data = self.transforms(data)
            
        if self.labels is not None:
            return (data, self.labels[index])
        else:
            return data
