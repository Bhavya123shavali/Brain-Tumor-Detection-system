"""
Main Pipeline - RAG-CNN Integrated Framework for Medical Diagnosis
Combines CNN image analysis with RAG-based knowledge retrieval

Key Integration (per UROP Project spec):
- CNN output embeddings are directly integrated into RAG retrieval pipeline
- Multimodal search combines text queries with CNN visual embeddings
- FAISS vector database handles both text and image feature alignment
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from cnn_model import BrainTumorCNN
from rag_module import MedicalVectorDB, initialize_medical_knowledge_base
from report_generator import MedicalReportGenerator
from config import VECTOR_DB_DIR, MODELS_DIR


class RAGCNNDiagnosticSystem:
    """
    Integrated RAG-CNN Framework for Automated Medical Diagnosis.
    
    This system combines:
    1. CNN (VGG16) for brain MRI image analysis
    2. FAISS-based RAG for medical literature retrieval
    3. Gemini API for comprehensive report generation
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        vector_db_path: Optional[str] = None,
        gemini_api_key: Optional[str] = None
    ):
        print("=" * 60)
        print("  RAG-CNN Medical Diagnostic System")
        print("  Initializing components...")
        print("=" * 60)
        
        # Initialize CNN Model
        print("\n[1/3] Loading CNN Model...")
        self.cnn_model = BrainTumorCNN(model_path)
        
        # Initialize Vector Database
        print("\n[2/3] Loading Medical Knowledge Base...")
        if vector_db_path and Path(vector_db_path).exists():
            self.vector_db = MedicalVectorDB(vector_db_path)
        else:
            # Check if default path exists
            default_db_path = VECTOR_DB_DIR / "medical_index"
            if default_db_path.exists():
                self.vector_db = MedicalVectorDB(default_db_path)
            else:
                print("  Creating new knowledge base with sample data...")
                self.vector_db = initialize_medical_knowledge_base()
        
        # Initialize Report Generator
        print("\n[3/3] Initializing Report Generator...")
        self.report_generator = MedicalReportGenerator(gemini_api_key)
        
        print("\n" + "=" * 60)
        print("  System Ready!")
        print("=" * 60 + "\n")
    
    def analyze_mri(
        self,
        image_path: str,
        patient_info: Optional[Dict] = None,
        generate_report: bool = True
    ) -> Dict[str, Any]:
        """
        Perform complete MRI analysis pipeline.
        
        Args:
            image_path: Path to MRI image file
            patient_info: Optional anonymized patient information
            generate_report: Whether to generate a detailed report
            
        Returns:
            Dictionary containing analysis results and report
        """
        print(f"\n📷 Analyzing MRI image: {image_path}")
        print("-" * 50)
        
        # Step 1: CNN Analysis
        print("\n🧠 Step 1: Running CNN Analysis...")
        cnn_results = self.cnn_model.get_diagnostic_features(image_path)
        
        print(f"  ✓ Classification: {cnn_results['classification']}")
        print(f"  ✓ Confidence: {cnn_results['confidence_score']}")
        
        # Step 2: Retrieve relevant literature using CNN embeddings + text query (Multimodal RAG)
        print("\n📚 Step 2: Retrieving Relevant Medical Literature (Multimodal RAG)...")
        search_query = self._build_search_query(cnn_results)
        
        # Get CNN embedding for multimodal retrieval
        cnn_embedding = cnn_results.get('embedding', None)
        
        # Use hybrid search combining text query with CNN embedding
        if cnn_embedding is not None:
            print("  → Using CNN embedding for enhanced retrieval")
            retrieved_docs = self.vector_db.search_with_embedding(
                query=search_query,
                image_embedding=cnn_embedding,
                alpha=0.7,  # 70% text, 30% image embedding influence
                top_k=5
            )
        else:
            retrieved_docs = self.vector_db.search(search_query)
        
        print(f"  ✓ Retrieved {len(retrieved_docs)} relevant documents")
        for i, doc in enumerate(retrieved_docs[:3]):
            topic = doc.get('metadata', {}).get('topic', 'N/A')
            print(f"    {i+1}. {topic} (relevance: {1/(1+doc['score']):.2f})")
        
        # Step 3: Generate Report
        report = None
        if generate_report:
            print("\n📋 Step 3: Generating Diagnostic Report...")
            report = self.report_generator.generate_report(
                cnn_results,
                retrieved_docs,
                patient_info
            )
            print("  ✓ Report generated successfully")
        
        # Compile results
        results = {
            "cnn_analysis": cnn_results,
            "retrieved_documents": retrieved_docs,
            "report": report,
            "summary": {
                "classification": cnn_results['classification'],
                "confidence": cnn_results['confidence_score'],
                "tumor_detected": "Tumor" in cnn_results['classification'],
                "documents_retrieved": len(retrieved_docs)
            }
        }
        
        print("\n" + "=" * 50)
        print("  Analysis Complete!")
        print("=" * 50)
        
        return results
    
    def _build_search_query(self, cnn_results: Dict) -> str:
        """Build search query based on CNN analysis results."""
        classification = cnn_results.get('classification', '')
        
        if 'Tumor' in classification:
            return "brain tumor diagnosis treatment MRI glioma meningioma symptoms prognosis"
        else:
            return "brain MRI normal findings neurological health screening"
    
    def add_medical_documents(self, documents: list, metadata_list: Optional[list] = None):
        """Add new documents to the medical knowledge base."""
        self.vector_db.add_documents(documents, metadata_list)
        print(f"Added {len(documents)} documents to knowledge base")
    
    def search_knowledge_base(self, query: str, top_k: int = 5) -> list:
        """Search the medical knowledge base."""
        return self.vector_db.search(query, top_k)
    
    def get_system_status(self) -> Dict:
        """Get current system status and statistics."""
        return {
            "cnn_model": {
                "architecture": "VGG16",
                "device": str(self.cnn_model.device),
                "classes": ["No Tumor", "Tumor Detected"],
                "is_trained": getattr(self.cnn_model, 'is_trained', False)
            },
            "vector_database": self.vector_db.get_stats(),
            "report_generator": {
                "model": "Gemini 1.5 Flash" if self.report_generator.model else "Fallback Template",
                "api_configured": self.report_generator.model is not None
            }
        }


def run_demo():
    """Run a demonstration of the RAG-CNN system."""
    print("\n" + "=" * 70)
    print("  RAG-CNN Medical Diagnostic System - Demo Mode")
    print("=" * 70 + "\n")
    
    # Initialize system
    system = RAGCNNDiagnosticSystem()
    
    # Print system status
    print("\n📊 System Status:")
    status = system.get_system_status()
    print(f"  CNN Model: {status['cnn_model']['architecture']} on {status['cnn_model']['device']}")
    print(f"  Knowledge Base: {status['vector_database']['total_documents']} documents")
    print(f"  Report Generator: {status['report_generator']['model']}")
    
    # Test knowledge base search
    print("\n" + "-" * 50)
    print("📚 Testing Knowledge Base Search")
    print("-" * 50)
    
    queries = [
        "What is glioblastoma?",
        "How is brain tumor treated?",
        "MRI imaging for brain tumor"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        results = system.search_knowledge_base(query, top_k=2)
        for i, res in enumerate(results):
            topic = res['metadata'].get('topic', 'N/A')
            print(f"  {i+1}. [{topic}] {res['document'][:100]}...")
    
    print("\n" + "=" * 70)
    print("  Demo Complete!")
    print("  To analyze real MRI images, use: system.analyze_mri('path/to/image.jpg')")
    print("=" * 70 + "\n")
    
    return system


if __name__ == "__main__":
    system = run_demo()
