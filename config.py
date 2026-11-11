"""
Configuration file for RAG-CNN Medical Diagnosis System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, VECTOR_DB_DIR]:
    dir_path.mkdir(exist_ok=True)

# Sub-directories
MRI_DATA_DIR = DATA_DIR / "mri_images"
MEDICAL_TEXTS_DIR = DATA_DIR / "medical_texts"
MRI_DATA_DIR.mkdir(exist_ok=True)
MEDICAL_TEXTS_DIR.mkdir(exist_ok=True)

# CNN Model Configuration
CNN_CONFIG = {
    "model_name": "vgg16",
    "num_classes": 2,  # tumor / no_tumor
    "input_size": (224, 224),
    "batch_size": 16,
    "epochs": 20,
    "learning_rate": 0.0001,
    "pretrained": True,
}

# FAISS Configuration
FAISS_CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2",  # sentence-transformers model
    "embedding_dimension": 384,
    "index_type": "IndexFlatL2",
    "top_k": 5,  # Number of documents to retrieve
}

# Gemini API Configuration
GEMINI_CONFIG = {
    "model_name": "gemini-2.0-flash",
    "temperature": 0.3,
    "max_output_tokens": 2048,
}

# API Key (set via environment variable for security)
# Create a .env file with: GOOGLE_API_KEY=your_api_key_here
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Image Classes
CLASS_LABELS = {
    0: "No Tumor",
    1: "Tumor Detected"
}

# Medical Report Template
REPORT_TEMPLATE = """
Based on the MRI scan analysis and retrieved medical literature, generate a comprehensive diagnostic report.

## Image Analysis Results:
{cnn_results}

## Retrieved Medical Context:
{retrieved_context}

Please provide:
1. Diagnostic Summary
2. Key Findings from Image Analysis
3. Relevant Medical Literature Insights
4. Recommended Next Steps
5. Important Disclaimers

Format the response as a professional medical report.
"""
