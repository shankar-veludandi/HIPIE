# -*- coding: utf-8 -*-
"""Baseline.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_yHBVtWGRX6ovFAbD_nUIXffIP6_egDz
"""

# Install FiftyOne and Other Dependencies
#!pip install fiftyone torch torchvision pycocotools

# Load the COCO Dataset with FiftyOne

import fiftyone as fo
import fiftyone.zoo as foz

# Load the COCO train split
train_dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections", "segmentations"],
    max_samples=None,  # You can adjust this based on your needs
    persistent=True
)

# Load the COCO validation split
val_dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections", "segmentations"],
    max_samples=None,  # You can adjust this based on your needs
    persistent=True
)

# Launch the FiftyOne app to visualize the datasets
#train_session = fo.launch_app(train_dataset)
#val_session = fo.launch_app(val_dataset)

# Define a Custom Dataset Class for COCO

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

class FiftyOneDataset(Dataset):
    def __init__(self, fiftyone_dataset, transform=None):
        self.dataset = fiftyone_dataset
        self.transform = transform
        self.samples = [s for s in self.dataset]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample.filepath).convert("RGB")
        target = torch.zeros(90)  # Assuming 90 classes

        for detection in sample.detections.detections:
            category_id = detection.label_id
            target[category_id - 1] = 1

        if self.transform:
            image = self.transform(image)

        return image, target

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the datasets
train_dataset = FiftyOneDataset(train_dataset, transform=transform)
val_dataset = FiftyOneDataset(val_dataset, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

import torch
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.backends.cudnn as cudnn

# Function to set the seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

# Set the seed
seed = 0
set_seed(seed)

# Define the model
class ResNetMultiLabel(nn.Module):
    def __init__(self, num_classes=90):
        super(ResNetMultiLabel, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        return self.sigmoid(x)

# Initialize the model, loss function, and optimizer
model = ResNetMultiLabel(num_classes=90)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training loop
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

# Evaluation loop
def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            predictions = (outputs > 0.5).float()
            all_targets.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)

    return val_loss / len(val_loader), all_targets, all_predictions

num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_targets, val_predictions = evaluate(model, val_loader, criterion, device)

    accuracy = accuracy_score(val_targets, val_predictions)
    precision = precision_score(val_targets, val_predictions, average='macro', zero_division=1)
    recall = recall_score(val_targets, val_predictions, average='macro', zero_division=1)
    f1 = f1_score(val_targets, val_predictions, average='macro', zero_division=1)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('-' * 50)

    # Log metrics to FiftyOne
    val_dataset.info[f"epoch_{epoch+1}"] = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
    val_dataset.save()

torch.save(model.state_dict(), 'resnet_multilabel.pth')

for sample, target, prediction in zip(val_dataset, val_targets, val_predictions):
    sample["ground_truth"] = fo.Classification(label=target)
    sample["prediction"] = fo.Classification(label=prediction)
    sample.save()

results = val_dataset.evaluate_classifications(
    "prediction",
    gt_field="ground_truth",
    eval_key="classification_eval",
)

print(results.metrics())

session = fo.launch_app(val_dataset)
