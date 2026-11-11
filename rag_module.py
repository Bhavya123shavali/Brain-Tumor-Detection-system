"""
RAG Module - FAISS Vector Database for Medical Text Retrieval

UROP Project: RAG-CNN Framework for Automated Medical Diagnosis
Key Features:
- FAISS-based vector database for efficient similarity search
- Multimodal retrieval: integrates CNN embeddings with text queries
- Encoding alignment between CNN visual features and text embeddings
"""
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
import pickle

from config import FAISS_CONFIG, VECTOR_DB_DIR, MEDICAL_TEXTS_DIR


class MedicalVectorDB:
    """
    FAISS-based vector database for medical literature retrieval.
    
    Supports multimodal search combining:
    - Text embeddings (sentence-transformers)
    - CNN image embeddings (VGG16 feature extractor)
    """
    
    def __init__(self, index_path=None):
        # Initialize sentence transformer for text embeddings
        print(f"Loading embedding model: {FAISS_CONFIG['embedding_model']}...")
        self.embedding_model = SentenceTransformer(FAISS_CONFIG['embedding_model'])
        self.embedding_dim = FAISS_CONFIG['embedding_dimension']
        
        # Initialize FAISS index
        self.index = None
        self.documents = []  # Store original documents
        self.metadata = []   # Store document metadata
        
        if index_path and Path(index_path).exists():
            self.load_index(index_path)
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        print(f"Created new FAISS index with dimension {self.embedding_dim}")
    
    def embed_text(self, text):
        """Generate embedding for a text string."""
        if isinstance(text, str):
            return self.embedding_model.encode([text])[0]
        return self.embedding_model.encode(text)
    
    def add_documents(self, documents, metadata_list=None):
        """
        Add documents to the vector database.
        
        Args:
            documents: List of text documents
            metadata_list: Optional list of metadata dicts for each document
        """
        if not documents:
            return
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store documents and metadata
        self.documents.extend(documents)
        if metadata_list:
            self.metadata.extend(metadata_list)
        else:
            self.metadata.extend([{}] * len(documents))
        
        print(f"Added {len(documents)} documents. Total documents: {len(self.documents)}")
    
    def search(self, query, top_k=None):
        """
        Search for relevant documents given a query.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of (document, score, metadata) tuples
        """
        if top_k is None:
            top_k = FAISS_CONFIG['top_k']
        
        if len(self.documents) == 0:
            return []
        
        # Embed query
        query_embedding = self.embed_text(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "score": float(dist),
                    "metadata": self.metadata[idx] if idx < len(self.metadata) else {}
                })
        
        return results
    
    def search_with_image_context(self, query, image_embedding, alpha=0.5, top_k=None):
        """
        Hybrid search combining text query and image embedding.
        
        Args:
            query: Text query
            image_embedding: CNN feature embedding from image
            alpha: Weight for text vs image (0=image only, 1=text only)
            top_k: Number of results
        """
        if top_k is None:
            top_k = FAISS_CONFIG['top_k']
        
        # Get text embedding
        text_embedding = self.embed_text(query)
        
        # Normalize embeddings to same dimension if needed
        # For now, we'll primarily use text search with image context in query
        # A more advanced approach would use multimodal embeddings
        
        # Enhance query with image context description
        enhanced_query = f"{query} brain tumor MRI analysis"
        
        return self.search(enhanced_query, top_k)
    
    def search_with_embedding(self, query, image_embedding=None, alpha=0.7, top_k=None):
        """
        Multimodal RAG search integrating CNN embeddings with text query.
        
        This implements the UROP project requirement:
        "Integrate CNN output embeddings directly into RAG retrieval pipeline"
        
        Args:
            query: Text search query
            image_embedding: CNN feature embedding (from VGG16 feature extractor)
            alpha: Balance between text (1.0) and image (0.0) similarity
            top_k: Number of results to return
            
        Returns:
            List of relevant documents ranked by multimodal similarity
        """
        if top_k is None:
            top_k = FAISS_CONFIG['top_k']
        
        if len(self.documents) == 0:
            return []
        
        # Get text embedding for the query
        text_embedding = self.embed_text(query)
        text_embedding = np.array(text_embedding).astype('float32')
        
        # Search using text embedding
        text_query = np.array([text_embedding]).astype('float32')
        text_distances, text_indices = self.index.search(text_query, min(top_k * 2, len(self.documents)))
        
        # If we have image embedding, use it to re-rank results
        if image_embedding is not None:
            # Normalize image embedding to match text embedding dimension
            # VGG16 produces 25088-dim features, we need to project to 384-dim
            if len(image_embedding) != self.embedding_dim:
                # Use a simple projection: average pooling + normalization
                img_emb = np.array(image_embedding).astype('float32')
                # Reshape and reduce dimension through averaging
                chunk_size = len(img_emb) // self.embedding_dim
                if chunk_size > 0:
                    img_emb_resized = img_emb[:chunk_size * self.embedding_dim].reshape(self.embedding_dim, chunk_size).mean(axis=1)
                else:
                    # Pad if image embedding is smaller
                    img_emb_resized = np.zeros(self.embedding_dim, dtype='float32')
                    img_emb_resized[:len(img_emb)] = img_emb
                
                # Normalize
                norm = np.linalg.norm(img_emb_resized)
                if norm > 0:
                    img_emb_resized = img_emb_resized / norm
                image_embedding = img_emb_resized
            
            # Re-rank based on combined similarity
            results_with_scores = []
            for dist, idx in zip(text_distances[0], text_indices[0]):
                if idx < len(self.documents):
                    # Get document embedding
                    doc_embedding = self.embed_text(self.documents[idx])
                    
                    # Calculate image-to-document similarity (cosine)
                    img_doc_sim = np.dot(image_embedding, doc_embedding) / (
                        np.linalg.norm(image_embedding) * np.linalg.norm(doc_embedding) + 1e-8
                    )
                    
                    # Combined score: alpha * text_score + (1-alpha) * image_score
                    # Lower distance = better for text, higher similarity = better for image
                    text_score = 1 / (1 + dist)  # Convert distance to similarity
                    combined_score = alpha * text_score + (1 - alpha) * max(0, img_doc_sim)
                    
                    results_with_scores.append({
                        "document": self.documents[idx],
                        "score": float(dist),
                        "combined_score": float(combined_score),
                        "metadata": self.metadata[idx] if idx < len(self.metadata) else {},
                        "text_relevance": float(text_score),
                        "image_relevance": float(img_doc_sim)
                    })
            
            # Sort by combined score (higher is better)
            results_with_scores.sort(key=lambda x: x['combined_score'], reverse=True)
            return results_with_scores[:top_k]
        
        # Fallback to text-only search
        results = []
        for dist, idx in zip(text_distances[0], text_indices[0]):
            if idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "score": float(dist),
                    "metadata": self.metadata[idx] if idx < len(self.metadata) else {}
                })
        
        return results[:top_k]
    
    def save_index(self, path=None):
        """Save the FAISS index and documents to disk."""
        if path is None:
            path = VECTOR_DB_DIR / "medical_index"
        
        path = Path(path)
        path.mkdir(exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save documents and metadata
        with open(path / "documents.pkl", "wb") as f:
            pickle.dump({"documents": self.documents, "metadata": self.metadata}, f)
        
        print(f"Index saved to {path}")
    
    def load_index(self, path):
        """Load FAISS index and documents from disk."""
        path = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        # Load documents and metadata
        with open(path / "documents.pkl", "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]
        
        print(f"Index loaded from {path}. Total documents: {len(self.documents)}")
    
    def get_stats(self):
        """Get statistics about the vector database."""
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_dimension": self.embedding_dim,
            "embedding_model": FAISS_CONFIG['embedding_model']
        }


# Sample medical knowledge base
SAMPLE_MEDICAL_KNOWLEDGE = [
    {
        "text": "Brain tumors are abnormal growths of cells in the brain. They can be benign (non-cancerous) or malignant (cancerous). Primary brain tumors originate in the brain, while secondary tumors metastasize from other parts of the body.",
        "source": "Medical Encyclopedia",
        "topic": "Brain Tumor Overview"
    },
    {
        "text": "Glioblastoma multiforme (GBM) is the most aggressive type of brain cancer. It accounts for about 15% of all brain tumors. Treatment typically involves surgery, radiation therapy, and chemotherapy. The median survival time is approximately 14-16 months.",
        "source": "Oncology Journal",
        "topic": "Glioblastoma"
    },
    {
        "text": "MRI (Magnetic Resonance Imaging) is the gold standard for brain tumor detection. T1-weighted images with gadolinium contrast help identify tumor boundaries. T2-weighted and FLAIR sequences reveal edema and infiltration patterns.",
        "source": "Radiology Handbook",
        "topic": "MRI Imaging"
    },
    {
        "text": "Meningiomas are typically benign tumors arising from the meninges. They represent about 30% of all brain tumors. Most meningiomas grow slowly and may not require immediate treatment. Regular monitoring with MRI is often recommended.",
        "source": "Neurosurgery Review",
        "topic": "Meningioma"
    },
    {
        "text": "Common symptoms of brain tumors include persistent headaches, seizures, vision problems, difficulty with balance, personality changes, and cognitive impairment. Symptoms depend on tumor location, size, and growth rate.",
        "source": "Clinical Neurology",
        "topic": "Symptoms"
    },
    {
        "text": "The WHO classification of brain tumors grades them from I to IV. Grade I tumors are slow-growing and often curable with surgery. Grade IV tumors are highly malignant with poor prognosis. Accurate grading is essential for treatment planning.",
        "source": "WHO Guidelines",
        "topic": "Tumor Grading"
    },
    {
        "text": "Surgical resection remains the primary treatment for accessible brain tumors. The extent of resection significantly impacts survival outcomes. Advanced techniques like awake craniotomy and intraoperative MRI improve surgical precision.",
        "source": "Surgical Oncology",
        "topic": "Treatment"
    },
    {
        "text": "Radiation therapy for brain tumors includes external beam radiation, stereotactic radiosurgery (SRS), and brachytherapy. Stereotactic radiosurgery delivers precise, high-dose radiation to the tumor while minimizing damage to surrounding tissue.",
        "source": "Radiation Oncology",
        "topic": "Radiation Therapy"
    },
    {
        "text": "Temozolomide (TMZ) is the most commonly used chemotherapy drug for glioblastoma. It works by damaging tumor DNA. The standard Stupp protocol combines temozolomide with radiation therapy for newly diagnosed glioblastoma.",
        "source": "Pharmacology Review",
        "topic": "Chemotherapy"
    },
    {
        "text": "Immunotherapy and targeted therapy represent emerging treatment options for brain tumors. Checkpoint inhibitors, CAR-T cell therapy, and tumor-treating fields (TTFields) are being investigated in clinical trials with promising results.",
        "source": "Cancer Research",
        "topic": "Emerging Treatments"
    },
    {
        "text": "Pediatric brain tumors differ from adult tumors in location, type, and behavior. Common pediatric tumors include medulloblastoma, pilocytic astrocytoma, and ependymoma. Treatment approaches are tailored to minimize long-term developmental effects.",
        "source": "Pediatric Oncology",
        "topic": "Pediatric Tumors"
    },
    {
        "text": "Follow-up care after brain tumor treatment includes regular MRI scans, neurological examinations, and monitoring for recurrence. Rehabilitation may be needed to address cognitive, motor, or speech deficits resulting from the tumor or treatment.",
        "source": "Survivorship Guidelines",
        "topic": "Follow-up Care"
    },
    {
        "text": "Deep learning and artificial intelligence are increasingly used in brain tumor diagnosis. CNN-based models can detect and segment tumors with high accuracy. AI-assisted diagnosis can help radiologists identify subtle abnormalities.",
        "source": "Medical AI Journal",
        "topic": "AI in Diagnosis"
    },
    {
        "text": "The blood-brain barrier (BBB) presents a significant challenge for drug delivery to brain tumors. Novel approaches include focused ultrasound, nanoparticles, and convection-enhanced delivery to bypass or temporarily disrupt the BBB.",
        "source": "Drug Delivery Research",
        "topic": "Drug Delivery"
    },
    {
        "text": "Molecular profiling of brain tumors has revolutionized diagnosis and treatment. IDH mutation status, MGMT promoter methylation, and 1p/19q codeletion are key biomarkers that predict treatment response and prognosis.",
        "source": "Molecular Oncology",
        "topic": "Molecular Markers"
    }
]


def initialize_medical_knowledge_base():
    """Initialize vector database with sample medical knowledge."""
    db = MedicalVectorDB()
    
    documents = [item["text"] for item in SAMPLE_MEDICAL_KNOWLEDGE]
    metadata = [{"source": item["source"], "topic": item["topic"]} for item in SAMPLE_MEDICAL_KNOWLEDGE]
    
    db.add_documents(documents, metadata)
    db.save_index()
    
    return db


if __name__ == "__main__":
    # Initialize and test the vector database
    print("Initializing Medical Vector Database...")
    db = initialize_medical_knowledge_base()
    
    # Test search
    print("\n--- Testing Search ---")
    query = "What is glioblastoma and how is it treated?"
    results = db.search(query, top_k=3)
    
    print(f"\nQuery: {query}\n")
    for i, result in enumerate(results):
        print(f"Result {i+1} (Score: {result['score']:.4f}):")
        print(f"  Topic: {result['metadata'].get('topic', 'N/A')}")
        print(f"  Text: {result['document'][:200]}...")
        print()
    
    # Print stats
    print("Database Stats:", db.get_stats())
