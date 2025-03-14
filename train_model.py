# -*- coding: utf-8 -*-
"""Inner_Liner_Permit_Classification (1).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1VwwX3K3u_y8g18eQpUmG4mgx10U-16jo
"""

import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import gdown

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_id = '1g7qNOZz9wC7OfOhcPqH1EZ5bk1UFGmlL'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'fairface-img-margin125-trainval.zip'  # Specify the output file name and extension
gdown.download(url, output, quiet=False)

!unzip fairface-img-margin125-trainval.zip

import gdown

file_id = '1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'fairface_label_train.csv'  # Specify the output file name and extension
gdown.download(url, output, quiet=False)

import gdown

file_id = '1wOdja-ezstMEp81tX1a-EYkFebev4h7D'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'fairface_label_val.csv'  # Specify the output file name and extension
gdown.download(url, output, quiet=False)

train_df = pd.read_csv('fairface_label_train.csv')
val_df = pd.read_csv('fairface_label_val.csv')

train_df.head()

exclude_races = ['Black', 'White', 'Latino_Hispanic', 'Middle Eastern']

train_df = pd.read_csv('fairface_label_train.csv')
train_filtered = train_df[~train_df['race'].isin(exclude_races)]  # <button class="citation-flag" data-index="5">
train_filtered.to_csv('train_filtered.csv', index=False)

val_df = pd.read_csv('fairface_label_val.csv')
val_filtered = val_df[~val_df['race'].isin(exclude_races)]
val_filtered.to_csv('val_filtered.csv', index=False)

class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Define label encoders
        self.age_map = {
            '0-2': 0, '3-9': 1, '10-19': 2, '20-29': 3,
            '30-39': 4, '40-49': 5, '50-59': 6, '60-69': 7,
            'more than 70': 8
        }
        self.gender_map = {"Male": 0, "Female": 1}
        self.race_map = {
            "East Asian": 0,
            "Indian": 1,
            "Southeast Asian": 2
        }

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx]['file'])
        image = Image.open(img_path).convert('RGB')

        # Encode labels as integers
        age = self.age_map[self.data.iloc[idx]['age']]
        gender = self.gender_map[self.data.iloc[idx]['gender']]
        race = self.race_map[self.data.iloc[idx]['race']]
        label = torch.tensor([age, gender, race])  # Convert to tensor

        if self.transform:
            image = self.transform(image)
        return image, label  # Return image and tensor label
    def __len__(self):
        # return the number of rows in the data
        return len(self.data)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageDataset(
    csv_file='train_filtered.csv',
    img_dir='',
    transform=transform
)

val_dataset = ImageDataset(
    csv_file='val_filtered.csv',
    img_dir='',
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # Load pretrained ResNet50
        self.base_model = models.resnet50(pretrained=True)

        # Add custom CNN layers on top
        self.additional_layers = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.fc_age = nn.Linear(2048, len(self.age_map))     # 9 age classes
        self.fc_gender = nn.Linear(2048, len(self.gender_map))  # 2 gender classes
        self.fc_race = nn.Linear(2048, len(self.race_map))   # 3 race classes


    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        x = self.additional_layers(x)
        x = x.mean([2, 3])  # Global average pooling

        age_out = self.fc_age(x)
        gender_out = self.fc_gender(x)
        race_out = self.fc_race(x)
        return age_out, gender_out, race_out

model = CustomModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    history = {'loss': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)  # Move labels to device

            # Forward pass
            age_out, gender_out, race_out = model(images)

            # Compute loss for each task
            age_loss = criterion(age_out, labels[:, 0])
            gender_loss = criterion(gender_out, labels[:, 1])
            race_loss = criterion(race_out, labels[:, 2])

            total_loss = age_loss + gender_loss + race_loss  # Combine losses

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

        epoch_loss = running_loss / len(train_loader)
        history['loss'].append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    return model, history

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_model, training_history = train_model(
    model=model,
    train_loader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    num_epochs=10
)

#model.load_state_dict(torch.load('trained_model.pth', map_location=torch.device('cpu')))
torch.save(model.state_dict(), 'trained_model.pth')

torch.save({
    'epoch': 10,  # current epoch number
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': criterion,  # or any other metrics you want to save
}, 'checkpoint.pth')

from PIL import Image
import torch

# 1. Load your sample image and convert it to RGB
img_path = 'val/3.jpg'  # Replace with your image path
img = Image.open(img_path).convert('RGB')

# 2. Apply the same transforms used during training
img = transform(img)  # 'transform' is defined in your training pipeline
img = img.unsqueeze(0)  # Add a batch dimension

# 3. Set the model to evaluation mode and move to the correct device
model.eval()
img = img.to(device)

# 4. Run the model in a no_grad block
with torch.no_grad():
    age_out, gender_out, race_out = model(img)

print("Model output:", age_out, gender_out, race_out)

def decode_predictions(age_map, gender_map, race_map, age_out, gender_out, race_out):
    age_pred = age_out.argmax(1).item()
    gender_pred = gender_out.argmax(1).item()
    race_pred = race_out.argmax(1).item()

    # Map integers back to categories
    age = list(age_map.keys())[age_pred]
    gender = list(gender_map.keys())[gender_pred]
    race = list(race_map.keys())[race_pred]
    return age, gender, race

age, gender, race = decode_predictions(train_dataset.age_map, train_dataset.gender_map, train_dataset.race_map, age_out, gender_out, race_out) # passed in the mapping dictionaries
print(f"Predicted: Age - {age}, Gender - {gender}, Race - {race}")

# probabilities = F.softmax(age_out, dim=1)
# labels = list(train_dataset.age_map.keys())
# for idx, prob in enumerate(probabilities[0]):
#     print(f"{labels[idx]}: {prob.item() * 100:.2f}%")

# probabilities = F.softmax(gender_out, dim=1)
# labels = list(train_dataset.gender_map.keys())
# for idx, prob in enumerate(probabilities[0]):
#     print(f"{labels[idx]}: {prob.item() * 100:.2f}%")

# probabilities = F.softmax(race_out, dim=1)
# labels = list(train_dataset.race_map.keys())
# for idx, prob in enumerate(probabilities[0]):
#     print(f"{labels[idx]}: {prob.item() * 100:.2f}%")


dummy_input = torch.randn(1, 3, 224, 224).to(device)
torch.onnx.export(
model,
dummy_input,
"model.onnx",
export_params=True,
opset_version=13,
input_names=['input'],
output_names=['age_out', 'gender_out', 'race_out'],
dynamic_axes=None
)
