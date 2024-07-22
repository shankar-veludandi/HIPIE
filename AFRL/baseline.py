import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights
from torchvision.datasets.coco import CocoDetection
import torch.backends.cudnn as cudnn
from PIL import Image
import numpy as np
import os
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
import csv

NUM_CLASSES = 90

# Custom Dataset Class
class CocoDetectionDataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.coco = CocoDetection(root=root, annFile=annFile)
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.coco[index] # Tuple unpacking, target holds the annotations for the objects present in the image at the index
        if self.transform:
            img = self.transform(img)

        # Convert target to multi-label binary vector
        labels = torch.zeros(NUM_CLASSES)

        # For each object in the image, the corr. index in labels is set to 1
        for obj in target:
            category_id = obj['category_id']
            labels[category_id - 1] = 1

        return img, labels

    def __len__(self):
        return len(self.coco)


# Define the model
class ResNetMultiLabel(nn.Module):
    def __init__(self):
        super(ResNetMultiLabel, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, NUM_CLASSES)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        return self.sigmoid(x)


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
    all_probabilities = []

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
            all_probabilities.append(outputs.cpu().numpy())

    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)
    all_probabilities = np.vstack(all_probabilities)

    return val_loss / len(val_loader), all_targets, all_predictions, all_probabilities


# Function to write metrics to the output file
def write_metrics(epoch, train_loss, val_loss, val_targets, val_predictions, output_file):
    accuracy = accuracy_score(val_targets, val_predictions)
    precision = precision_score(val_targets, val_predictions, average='macro', zero_division=1)
    recall = recall_score(val_targets, val_predictions, average='macro', zero_division=1)
    f1 = f1_score(val_targets, val_predictions, average='macro', zero_division=1)

    output_file.write(f'{epoch}\t{train_loss:.4f}\t{val_loss:.4f}\t{accuracy:.4f}\t{precision:.4f}\t{recall:.4f}\t{f1:.4f}\n')


# Function to write the confusion matrix to the output file
def write_confusion_matrix(val_targets, val_predictions, output_file):
    output_file.write('# Confusion Matrix\n')
    summed_confusion_matrix = np.zeros((2, 2))
    for class_idx in range(val_targets.shape[1]):
        class_confusion_matrix = confusion_matrix(val_targets[:, class_idx], val_predictions[:, class_idx], labels=[0, 1])
        summed_confusion_matrix += class_confusion_matrix

    output_file.write("Class_0_TP\tClass_0_FP\tClass_0_FN\tClass_0_TN\n")
    for row in summed_confusion_matrix:
        output_file.write("\t".join(map(str, row)) + "\n")


# Function to write the ROC curve to the output file
def write_roc_curve(val_targets, val_probabilities, output_file):
    output_file.write('# ROC Curve\n')
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(val_targets.shape[1]):
        if len(np.unique(val_targets[:, i])) == 2:
            fpr[i], tpr[i], _ = roc_curve(val_targets[:, i], val_probabilities[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in fpr]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in fpr:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(fpr)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    output_file.write("Class\tFPR\tTPR\n")
    for i in fpr:
        for fp, tp in zip(fpr[i], tpr[i]):
            output_file.write(f"{i}\t{fp:.4f}\t{tp:.4f}\n")


def main(seed):

  # Define transforms
  transform = transforms.Compose([
      transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
      transforms.ToTensor(),  # Convert images to PyTorch tensors
  ])

  # Load the COCO dataset
  train_dataset = CocoDetectionDataset(root='/content/coco/train2017',
                                        annFile='/content/coco/annotations/instances_train2017.json',
                                        transform=transform)

  val_dataset = CocoDetectionDataset(root='/content/coco/val2017',
                                      annFile='/content/coco/annotations/instances_val2017.json',
                                      transform=transform)


  # Define data loaders
  train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
  val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

  # set seed
  set_seed(seed)

  # Initialize the model, loss function, and optimizer
  model = ResNetMultiLabel()
  criterion = nn.BCELoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  # Move the model to GPU if available
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if device.type == 'cpu':
      raise ValueError("CUDA not available")
  model = model.to(device)

  # Train the model
  num_epochs = 15
  patience = 5  # Number of epochs with no improvement after which training will be stopped
  best_val_loss = float('inf')
  epochs_without_improvement = 0

  train_losses = []
  val_losses = []

  # Open a file to write the output
  with open("training_output.tsv", "a") as output_file:
    output_file.write(f'# Seed: {seed}\n')
    output_file.write('# Metrics\n')
    output_file.write('Epoch\tTrain_Loss\tVal_Loss\tAccuracy\tPrecision\tRecall\tF1_Score\n')

    for epoch in range(num_epochs):
      train_loss = train(model, train_loader, criterion, optimizer, device)
      val_loss, val_targets, val_predictions, val_probabilities = evaluate(model, val_loader, criterion, device)
      train_losses.append(train_loss)
      val_losses.append(val_loss)

      write_metrics(epoch, train_loss, val_loss, val_targets, val_predictions, output_file)

      # Early stopping logic
      if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        # Save the best model
        torch.save(model.state_dict(), 'baseline.pth')
      else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
          output_file.write("Early stopping due to no improvement in validation loss")
          break

    write_confusion_matrix(val_targets, val_predictions, output_file)
    write_roc_curve(val_targets, val_probabilities, output_file)


seeds = [0, 1, 42, 123, 1024]
for seed in seeds:
  main(seed)
