"""
Download and prepare brain MRI dataset for training
Downloads from Kaggle Brain MRI dataset
"""
import os
import urllib.request
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm

from config import MRI_DATA_DIR, MODELS_DIR


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_sample_dataset():
    """
    Download sample brain MRI images for testing.
    Uses a small subset for quick testing.
    """
    print("=" * 60)
    print("  Brain MRI Dataset Downloader")
    print("=" * 60)
    
    # Create directories
    tumor_dir = MRI_DATA_DIR / "tumor"
    no_tumor_dir = MRI_DATA_DIR / "no_tumor"
    tumor_dir.mkdir(parents=True, exist_ok=True)
    no_tumor_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists
    tumor_count = len(list(tumor_dir.glob("*.jpg"))) + len(list(tumor_dir.glob("*.png")))
    no_tumor_count = len(list(no_tumor_dir.glob("*.jpg"))) + len(list(no_tumor_dir.glob("*.png")))
    
    if tumor_count > 0 and no_tumor_count > 0:
        print(f"\n✓ Dataset already exists!")
        print(f"  Tumor images: {tumor_count}")
        print(f"  No tumor images: {no_tumor_count}")
        return True
    
    print("\n📥 To get the dataset, please:")
    print("1. Go to: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection")
    print("2. Download the dataset")
    print("3. Extract and place images in:")
    print(f"   - Tumor images: {tumor_dir}")
    print(f"   - Normal images: {no_tumor_dir}")
    
    print("\n📝 Alternatively, you can use any brain MRI images:")
    print("   - JPEG or PNG format")
    print("   - Place tumor images in 'tumor' folder")
    print("   - Place healthy images in 'no_tumor' folder")
    
    return False


def create_sample_images():
    """
    Create synthetic sample images for testing purposes.
    These are NOT real MRI images - just for testing the pipeline.
    """
    import numpy as np
    from PIL import Image
    
    print("\n🔧 Creating synthetic test images (for pipeline testing only)...")
    
    tumor_dir = MRI_DATA_DIR / "tumor"
    no_tumor_dir = MRI_DATA_DIR / "no_tumor"
    tumor_dir.mkdir(parents=True, exist_ok=True)
    no_tumor_dir.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic "tumor" images (darker with a bright spot)
    for i in range(20):
        # Create a dark brain-like background with a bright tumor spot
        img = np.random.randint(20, 60, (224, 224, 3), dtype=np.uint8)
        
        # Add a bright spot (simulated tumor)
        cx, cy = np.random.randint(60, 164, 2)
        radius = np.random.randint(15, 40)
        y, x = np.ogrid[:224, :224]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        img[mask] = np.random.randint(180, 255, 3)
        
        # Add some brain-like structure
        for _ in range(5):
            sx, sy = np.random.randint(20, 200, 2)
            sr = np.random.randint(30, 60)
            smask = (x - sx)**2 + (y - sy)**2 <= sr**2
            img[smask] = np.clip(img[smask] + np.random.randint(10, 30), 0, 255)
        
        pil_img = Image.fromarray(img)
        pil_img.save(tumor_dir / f"tumor_sample_{i+1:03d}.jpg")
    
    # Create synthetic "no tumor" images (uniform gray matter)
    for i in range(20):
        # Create a uniform brain-like image without bright spots
        img = np.random.randint(40, 80, (224, 224, 3), dtype=np.uint8)
        
        # Add some brain-like structure
        for _ in range(8):
            sx, sy = np.random.randint(20, 200, 2)
            sr = np.random.randint(30, 70)
            smask = (x - sx)**2 + (y - sy)**2 <= sr**2
            img[smask] = np.clip(img[smask] + np.random.randint(5, 20), 0, 120)
        
        pil_img = Image.fromarray(img)
        pil_img.save(no_tumor_dir / f"normal_sample_{i+1:03d}.jpg")
    
    print(f"  ✓ Created 20 synthetic tumor images in {tumor_dir}")
    print(f"  ✓ Created 20 synthetic normal images in {no_tumor_dir}")
    print("\n⚠️  Note: These are synthetic images for testing only!")
    print("   For real diagnosis, use actual medical MRI datasets.")
    
    return True


if __name__ == "__main__":
    # First try to check for real dataset
    has_data = download_sample_dataset()
    
    if not has_data:
        print("\n" + "=" * 60)
        response = input("Create synthetic test images? (y/n): ").strip().lower()
        if response == 'y':
            create_sample_images()
            print("\n✓ Synthetic dataset created successfully!")
            print("Now run: python train_cnn.py")
