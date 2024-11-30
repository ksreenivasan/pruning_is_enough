import joblib
import cv2
import os
import time
import random
import pretrainedmodels
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

'''SEED Everything'''
def seed_everything(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True # keep True if all the input have same size.
SEED=42
seed_everything(SEED=SEED)
'''SEED Everything'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # GPU
epochs = 5 # Number of epochs
BS = 16 # Batch size


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

train_data = CustomDataset(x_train, y_train, train_transforms)
val_data = CustomDataset(x_val, y_val, val_transform)
test_data = CustomDataset(x_test, y_test, val_transform)       

trainLoader = DataLoader(train_data, batch_size=BS, shuffle=True, num_workers=4)
valLoader = DataLoader(val_data, batch_size=BS, shuffle=True, num_workers=4)
testLoader = DataLoader(test_data, batch_size=BS, shuffle=True, num_workers=4) 


class ResNet34(nn.Module):
    def __init__(self, pretrained):
        super(ResNet34, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained = None)
        # change the classification layer
        self.l0= nn.Linear(512, len(lb.classes_))
        self.dropout = nn.Dropout2d(0.4)
        
    def forward(self, x):
        # get the batch size only, ignore(c, h, w)
        batch, _, _, _ = x.size()
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        x = self.dropout(x)
        l0 = self.l0(x)
        return l0

class ResNet50(nn.Module):
    def __init__(self, pretrained):
        super(ResNet50, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained = None)
        # change the classification layer
        self.l0= nn.Linear(2048, len(lb.classes_))
        self.dropout = nn.Dropout2d(0.4)
        
    def forward(self, x):
        # get the batch size only, ignore(c, h, w)
        batch, _, _, _ = x.size()
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        x = self.dropout(x)
        l0 = self.l0(x)
        return l0

#model = ResNet34(pretrained=True).to(device)
model = ResNet50(pretrained=True).to(device)
print(model)

# loss function
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr = 1e-4)

# training function
#train_loss , train_accuracy = [], []
def train(model, trainLoader):
    model.train()
    running_loss = 0.0
    running_correct = 0
    for batch_idx, data in enumerate(trainLoader):
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        #loss = criterion(outputs, torch.max(target, 1)[1])
        loss = criterion(outputs, target)
        running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        #running_correct += (preds == torch.max(target, 1)[1]).sum().item()
        running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
        if batch_idx % 40 == 0:
            print("batch-[{}/{}] Loss: {}".format(batch_idx, len(trainLoader), loss.item()))
        
    loss = running_loss/len(trainLoader.dataset)
    accuracy = 100. * running_correct/len(trainLoader.dataset)
    
    print(f"Train Loss: {loss:.4f}, Train Acc: {accuracy:.2f}")
    return loss, accuracy

#validation function
def validate(model, dataloader):
    print('Validating')
    model.eval()
    running_loss = 0.0
    running_correct = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)
            #loss = criterion(outputs, torch.max(target, 1)[1])
            loss = criterion(outputs, target)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            #running_correct += (preds == torch.max(target, 1)[1]).sum().item()
            running_correct += (preds == target).sum().item()

        loss = running_loss/len(dataloader.dataset)
        accuracy = 100. * running_correct/len(dataloader.dataset)
        print(f'Val Loss: {loss:.4f}, Val Acc: {accuracy:.2f}')
        
        return loss, accuracy

def test(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testLoader:
            inputs, target = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == torch.max(target, 1)[1]).sum().item()
    return correct, total

if __name__ == "__main__":
    train_loss, train_accuracy = [], []
    val_loss, val_accuracy = [], []
    print(f"Training on {len(train_data)} examples, validating on {len(val_data)} examples...")
    start = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_accuracy = train(model, trainLoader)
        val_epoch_loss, val_epoch_accuracy = validate(model, valLoader)
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)
    end = time.time()
    print((end-start)/60, 'minutes')
