import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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
import matplotlib.pyplot as plt
import seaborn as sns

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


# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize images to 224x224 pixels
    transforms.ToTensor(), # Convert images to PyTorch tensors and scale pixel values to [0,1]
])


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


# Calculate metrics
def calc_metrics(val_targets, val_predictions, output_file):
  accuracy = accuracy_score(val_targets, val_predictions)
  precision = precision_score(val_targets, val_predictions, average='macro', zero_division=1)
  recall = recall_score(val_targets, val_predictions, average='macro', zero_division=1)
  f1 = f1_score(val_targets, val_predictions, average='macro', zero_division=1)

  output_file.write(f'Accuracy: {accuracy:.4f}')
  output_file.write(f'Precision: {precision:.4f}')
  output_file.write(f'Recall: {recall:.4f}')
  output_file.write(f'F1 Score: {f1:.4f}')
  output_file.write('-' * 50)


# Plot confusion matrix
def plot_confusion_matrix(val_targets, val_predictions):
  summed_confusion_matrix = np.zeros((2, 2)) # Calculate and plot summed confusion matrix for all classes

  for class_idx in range(NUM_CLASSES):
      class_confusion_matrix = confusion_matrix(val_targets[:, class_idx], val_predictions[:, class_idx], labels=[0, 1])
      summed_confusion_matrix += class_confusion_matrix

  print("Summed Confusion Matrix for All Classes:")
  print(summed_confusion_matrix)

  plt.figure(figsize=(6, 4))
  sns.heatmap(summed_confusion_matrix, annot=True, fmt=".2f", cmap="Blues")
  plt.title('Summed Confusion Matrix for All Classes')
  plt.xlabel('Predicted Labels')
  plt.ylabel('True Labels')
  plt.show()


# Plot ROC curve
def plot_roc_curve(val_targets, val_probabilities):
  fpr = dict()
  tpr = dict()
  roc_auc = dict()

  for i in range(NUM_CLASSES):
      if len(np.unique(val_targets[:, i])) == 2:  # Check if there are both positive and negative samples
          fpr[i], tpr[i], _ = roc_curve(val_targets[:, i], val_probabilities[:, i])
          roc_auc[i] = auc(fpr[i], tpr[i])
      else:
          print(f"Class {i} is skipped because it has only one class present in y_true")

  # Compute macro-average ROC curve and AUC
  all_fpr = np.unique(np.concatenate([fpr[i] for i in fpr]))
  mean_tpr = np.zeros_like(all_fpr)

  for i in fpr:
      mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

  mean_tpr /= len(fpr)

  fpr["macro"] = all_fpr
  tpr["macro"] = mean_tpr
  roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

  # Plot macro-average ROC curve
  plt.figure()
  plt.plot(fpr["macro"], tpr["macro"],
          label=f'Macro-average ROC curve (area = {roc_auc["macro"]:.2f})',
          color='navy', linestyle=':', linewidth=4)

  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Macro-Average ROC Curve')
  plt.legend(loc="lower right")
  plt.show()

# Plot the training and validation loss to find convergence
def plot_losses(num_epochs, train_losses, val_losses):
  plt.figure(figsize=(10, 5))
  plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
  plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Training and Validation Loss Over Epochs')
  plt.legend()
  plt.grid(True)
  plt.show()

def main(world_size, rank): 

  if not torch.cuda.is_available():
      raise ValueError("CUDA not available")

  # Initialize the process group
  dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

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

  # Define data loaders with DistributedSampler
  train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
  val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

  # Define data loaders
  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
  val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8)

  # Rerun model on seeds 0, 1, 42, 123, 1024

  # Set the seed
  seed = 0
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

  # Wrap the model with DistributedDataParallel
  model = DDP(model, device_ids=[rank])

  # Train the model
  num_epochs = 100
  patience = 5  # Number of epochs with no improvement after which training will be stopped
  best_val_loss = float('inf')
  epochs_without_improvement = 0

  train_losses = []
  val_losses = []

  # Open a file to write the output
  output_file = open("training_output.txt", "w")

  for epoch in range(num_epochs):
      train_loss = train(model, train_loader, criterion, optimizer, device)
      val_loss, val_targets, val_predictions, val_probabilities = evaluate(model, val_loader, criterion, device)
      output_file.write(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
      train_losses.append(train_loss)
      val_losses.append(val_loss)

      calc_metrics(val_targets, val_predictions, output_file)

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

      # Plot confusion matrix and ROC curve at the end of each epoch
      #plot_confusion_matrix(val_targets, val_predictions)
      #plot_roc_curve(val_targets, val_probabilities)

  plot_losses(num_epochs, train_losses, val_losses)

  dist.destroy_process_group()

  # Save the model
  torch.save(model.state_dict(), 'resnet_multilabel.pth')

world_size = 4
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '????'
mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
