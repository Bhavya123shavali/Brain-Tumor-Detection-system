"""
CNN Model Module for Brain MRI Tumor Classification
Uses VGG16 as the base model with custom classification head
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path

from config import CNN_CONFIG, CLASS_LABELS, MODELS_DIR


class BrainTumorCNN:
    """
    CNN model for brain tumor detection using VGG16 architecture.
    """
    
    def __init__(self, model_path=None):
        # Force CUDA if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print(f"Using device: CPU")
        
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Track if model is trained
        self.is_trained = False
        
        # Try to load model from various paths
        model_loaded = False
        
        # Priority: provided path > models folder > default
        possible_paths = []
        if model_path:
            possible_paths.append(Path(model_path))
        possible_paths.extend([
            MODELS_DIR / "brain_tumor_cnn.pth",
            MODELS_DIR / "best_brain_tumor_cnn.pth",
            Path("brain_tumor_cnn.pth"),
        ])
        
        for path in possible_paths:
            if path.exists():
                try:
                    self.load_model(path)
                    self.is_trained = True
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
        
        if not model_loaded:
            print("⚠️ No trained model found. Using pretrained VGG16 (untrained for tumor detection).")
            print("   Train a model using the Colab notebook: RAG_CNN_Training_Colab.ipynb")
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize(CNN_CONFIG["input_size"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Feature extractor (for RAG integration)
        self.feature_extractor = self._build_feature_extractor()
    
    def _build_model(self):
        """Build VGG16 model with custom classifier for tumor detection."""
        # Load pretrained VGG16
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if CNN_CONFIG["pretrained"] else None)
        
        # Freeze early layers (optional - for transfer learning)
        for param in model.features[:20].parameters():
            param.requires_grad = False
        
        # Replace classifier for binary classification
        num_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, CNN_CONFIG["num_classes"]),
        )
        
        return model
    
    def _build_feature_extractor(self):
        """Build feature extractor for CNN embeddings (used in RAG integration)."""
        # Use the features part of VGG16 + adaptive pooling
        feature_model = nn.Sequential(
            self.model.features,
            self.model.avgpool,
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512)  # Project to 512-dim embedding
        )
        feature_model.to(self.device)
        feature_model.eval()
        return feature_model
    
    def preprocess_image(self, image_path):
        """Load and preprocess an MRI image."""
        if isinstance(image_path, str) or isinstance(image_path, Path):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
        
        return self.transform(image).unsqueeze(0)
    
    def predict(self, image_path):
        """
        Predict tumor presence in MRI image.
        
        Returns:
            dict: Contains prediction, confidence, and feature embedding
        """
        self.model.eval()
        
        # Preprocess image
        image_tensor = self.preprocess_image(image_path).to(self.device)
        
        with torch.no_grad():
            # Get prediction
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Get feature embedding for RAG
            features = self.feature_extractor(image_tensor)
            embedding = features.cpu().numpy().flatten()
        
        return {
            "prediction": CLASS_LABELS[predicted_class],
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": {
                CLASS_LABELS[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            },
            "embedding": embedding,
            "tumor_detected": predicted_class == 1
        }
    
    def extract_features(self, image_path):
        """Extract CNN feature embedding for an image (for RAG integration)."""
        self.model.eval()
        image_tensor = self.preprocess_image(image_path).to(self.device)
        
        with torch.no_grad():
            features = self.feature_extractor(image_tensor)
        
        return features.cpu().numpy().flatten()
    
    def get_diagnostic_features(self, image_path):
        """
        Extract detailed diagnostic features from MRI image.
        Returns human-readable analysis of the image.
        """
        result = self.predict(image_path)
        
        # Create detailed diagnostic summary
        diagnostic_info = {
            "classification": result["prediction"],
            "confidence_score": f"{result['confidence']*100:.2f}%",
            "tumor_probability": f"{result['probabilities'].get('Tumor Detected', 0)*100:.2f}%",
            "normal_probability": f"{result['probabilities'].get('No Tumor', 0)*100:.2f}%",
            "analysis_summary": self._generate_analysis_summary(result),
            "embedding": result["embedding"]
        }
        
        return diagnostic_info
    
    def _generate_analysis_summary(self, result):
        """Generate a human-readable summary of the analysis."""
        if result["tumor_detected"]:
            if result["confidence"] > 0.9:
                return "High-confidence detection of abnormal mass in brain MRI. Immediate specialist consultation recommended."
            elif result["confidence"] > 0.7:
                return "Moderate-confidence detection of potential abnormality. Further imaging and specialist review advised."
            else:
                return "Low-confidence detection of possible abnormality. Additional scans recommended for confirmation."
        else:
            if result["confidence"] > 0.9:
                return "Brain MRI appears normal with high confidence. No visible abnormalities detected."
            elif result["confidence"] > 0.7:
                return "Brain MRI appears normal. Routine follow-up recommended."
            else:
                return "No definitive abnormality detected, but confidence is low. Consider additional imaging."
    
    def save_model(self, path=None):
        """Save the trained model."""
        if path is None:
            path = MODELS_DIR / "brain_tumor_cnn.pth"
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model - handles both Colab and local formats."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'vgg16_state_dict' in checkpoint:
                # Full checkpoint from Colab (brain_tumor_cnn_full.pth)
                self.model.load_state_dict(checkpoint['vgg16_state_dict'])
                print(f"✅ Model loaded from {path} (Colab full checkpoint)")
                if 'best_val_acc' in checkpoint:
                    print(f"   Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
            elif 'model_state_dict' in checkpoint:
                # Alternative format
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Model loaded from {path} (model_state_dict)")
            elif 'features.0.weight' in checkpoint:
                # Direct VGG16 state dict (brain_tumor_cnn.pth from Colab)
                self.model.load_state_dict(checkpoint)
                print(f"✅ Model loaded from {path} (VGG16 state dict)")
            else:
                # Try loading as full model state dict
                self.model.load_state_dict(checkpoint)
                print(f"✅ Model loaded from {path}")
        else:
            # OrderedDict - direct state dict
            self.model.load_state_dict(checkpoint)
            print(f"✅ Model loaded from {path} (state dict)")


def create_sample_model():
    """Create and return a sample CNN model for testing."""
    model = BrainTumorCNN()
    print("Brain Tumor CNN model created successfully!")
    print(f"Model architecture: VGG16 with custom classifier")
    print(f"Number of classes: {CNN_CONFIG['num_classes']}")
    print(f"Input size: {CNN_CONFIG['input_size']}")
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_sample_model()
    
    # Print model summary
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
