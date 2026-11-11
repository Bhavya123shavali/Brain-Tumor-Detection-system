# 🧠 NeuroRAG: CNN + RAG Framework for Brain MRI Analysis

---

## 🚀 Overview

NeuroRAG is an AI-powered medical diagnostic system that combines **Convolutional Neural Networks (CNN)** with **Retrieval-Augmented Generation (RAG)** to analyze brain MRI scans and generate intelligent, evidence-based medical reports.

This project integrates computer vision and natural language processing to detect tumors and provide meaningful medical insights.

---

## 🎯 Key Features

- 🧠 CNN-based MRI Classification using VGG16  
- 📊 86.27% accuracy in tumor detection  
- 🔎 FAISS vector database for medical knowledge retrieval  
- 🤖 RAG-based report generation using Gemini API  
- 💬 Conversational chatbot restricted to uploaded documents  
- 🌐 Streamlit web interface  
- 🔗 Multimodal pipeline (image + text understanding)  

---

## 🧠 System Architecture

```
User Interface (Streamlit)
	│
	▼
MRI Image Input ───► CNN (VGG16)
	│
	▼
Feature Embeddings
	│
	▼
FAISS Vector Database
	│
	▼
Retrieved Context
	│
	▼
Gemini API (RAG)
	│
	▼
Diagnostic Report
```


---

## 📁 Project Structure

```
.
├── app.py
├── config.py
├── cnn_model.py
├── rag_module.py
├── report_generator.py
├── main_pipeline.py
├── train_cnn.py
├── download_dataset.py
├── requirements.txt
├── .env.example
├── models/
├── vector_db/
├── data/
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/neuro-rag-brain-tumor-detection.git
cd neuro-rag-brain-tumor-detection
```

### 2. Setup Virtual Environment
```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
```bash
cp .env.example .env
# Edit .env and set:
GOOGLE_API_KEY=your_api_key_here
```

### 5. Train Model (Optional)
Run `train_cnn.py` OR use Colab notebook.
Save model in:
`models/brain_tumor_cnn.pth`


### 6. Run the App
```bash
streamlit run app.py
```
Open in browser: [http://localhost:8501](http://localhost:8501)

---

## 📊 Model Performance

- **Architecture:** VGG16
- **Accuracy:** 86.27%
- **Classes:** Tumor / No Tumor
- **Training:** 10 epochs

---

## 📝 Usage

**MRI Analysis**
- Upload MRI image
- Get prediction
- Generate report

**Knowledge Base Chat**
- Upload PDFs
- Ask questions
- Get AI responses

---

## 🔧 Tech Stack

- Python
- Streamlit
- PyTorch
- FAISS
- Gemini API
- NumPy, Pandas

---

## 💡 Future Improvements

- Improve accuracy
- Multi-class tumor detection
- Cloud deployment
- Explainable AI (Grad-CAM)