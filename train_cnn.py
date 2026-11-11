"""
CNN Training Module for Brain Tumor Classification
Trains VGG16 model on brain MRI dataset
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from cnn_model import BrainTumorCNN
from config import CNN_CONFIG, MRI_DATA_DIR, MODELS_DIR


class BrainMRIDataset(Dataset):
    """
    Custom Dataset for Brain MRI images.
    
    Expected folder structure:
    data/mri_images/
        ├── tumor/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        └── no_tumor/
            ├── image1.jpg
            ├── image2.jpg
            └── ...
    """
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {"no_tumor": 0, "tumor": 1}
        
        # Load all image paths and labels
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.[jJ][pP][gG]"):
                    self.samples.append((str(img_path), class_idx))
                for img_path in class_dir.glob("*.[pP][nN][gG]"):
                    self.samples.append((str(img_path), class_idx))
        
        print(f"Loaded {len(self.samples)} images from {root_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CNNTrainer:
    """
    Trainer class for the Brain Tumor CNN model.
    """
    
    def __init__(self, data_dir=None, model=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training device: {self.device}")
        
        self.data_dir = data_dir or MRI_DATA_DIR
        self.model = model or BrainTumorCNN()
        self.model.model.to(self.device)
        
        # Training transforms with data augmentation
        self.train_transform = transforms.Compose([
            transforms.Resize(CNN_CONFIG["input_size"]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Validation transforms (no augmentation)
        self.val_transform = transforms.Compose([
            transforms.Resize(CNN_CONFIG["input_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.model.parameters(),
            lr=CNN_CONFIG["learning_rate"]
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }
    
    def prepare_data(self, train_split=0.8):
        """Prepare training and validation datasets."""
        # Create full dataset
        full_dataset = BrainMRIDataset(self.data_dir, transform=self.train_transform)
        
        if len(full_dataset) == 0:
            raise ValueError(f"No images found in {self.data_dir}. Please add images to tumor/ and no_tumor/ folders.")
        
        # Split into train and validation
        train_size = int(train_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Update validation dataset transform
        val_dataset.dataset.transform = self.val_transform
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=CNN_CONFIG["batch_size"],
            shuffle=True,
            num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=CNN_CONFIG["batch_size"],
            shuffle=False,
            num_workers=0
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        return self.train_loader, self.val_loader
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model."""
        self.model.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, epochs=None, save_best=True):
        """
        Full training loop.
        
        Args:
            epochs: Number of training epochs
            save_best: Whether to save the best model
        """
        epochs = epochs or CNN_CONFIG["epochs"]
        best_val_loss = float('inf')
        
        print(f"\nStarting training for {epochs} epochs...")
        print("-" * 50)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.save_model(MODELS_DIR / "best_brain_tumor_cnn.pth")
                print("  ✓ Saved best model!")
        
        print("\n" + "-" * 50)
        print("Training complete!")
        
        return self.history
    
    def plot_history(self, save_path=None):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.history["train_loss"], label="Train Loss")
        ax1.plot(self.history["val_loss"], label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.history["train_acc"], label="Train Acc")
        ax2.plot(self.history["val_acc"], label="Val Acc")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Training and Validation Accuracy")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training history saved to {save_path}")
        
        plt.show()


def create_sample_data_structure():
    """Create the expected folder structure for training data."""
    tumor_dir = MRI_DATA_DIR / "tumor"
    no_tumor_dir = MRI_DATA_DIR / "no_tumor"
    
    tumor_dir.mkdir(parents=True, exist_ok=True)
    no_tumor_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"""
Data directory structure created at: {MRI_DATA_DIR}

Expected structure:
{MRI_DATA_DIR}/
├── tumor/
│   ├── tumor_image1.jpg
│   ├── tumor_image2.jpg
│   └── ...
└── no_tumor/
    ├── normal_image1.jpg
    ├── normal_image2.jpg
    └── ...

Please add your MRI images to the appropriate folders before training.

Recommended datasets:
1. BraTS (Brain Tumor Segmentation): https://www.med.upenn.edu/cbica/brats/
2. Brain Tumor Dataset on Kaggle: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
""")


if __name__ == "__main__":
    # Create folder structure
    create_sample_data_structure()
    
    print("\nTo train the model, add MRI images to the data folders and run:")
    print("  trainer = CNNTrainer()")
    print("  trainer.prepare_data()")
    print("  trainer.train(epochs=20)")
