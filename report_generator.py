"""
Report Generator Module using Gemini API
Generates comprehensive medical diagnostic reports
"""
import os
from typing import Dict, List, Optional
import google.generativeai as genai
from config import GEMINI_CONFIG, GOOGLE_API_KEY, REPORT_TEMPLATE


class MedicalReportGenerator:
    """
    Generates medical diagnostic reports using Google's Gemini API.
    Combines CNN analysis results with retrieved medical context.
    """
    
    def __init__(self, api_key=None):
        self.api_key = api_key or GOOGLE_API_KEY
        
        if not self.api_key:
            print("Warning: No API key provided. Set GOOGLE_API_KEY environment variable.")
            print("Report generation will use a template-based fallback.")
            self.model = None
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(GEMINI_CONFIG['model_name'])
            print(f"Initialized Gemini model: {GEMINI_CONFIG['model_name']}")
    
    def generate_report(
        self,
        cnn_results: Dict,
        retrieved_documents: List[Dict],
        patient_info: Optional[Dict] = None
    ) -> str:
        """
        Generate a comprehensive medical diagnostic report.
        
        Args:
            cnn_results: Dictionary containing CNN analysis results
            retrieved_documents: List of relevant medical documents from RAG
            patient_info: Optional patient information (anonymized)
            
        Returns:
            Formatted medical report string
        """
        # Format CNN results for the prompt
        cnn_summary = self._format_cnn_results(cnn_results)
        
        # Format retrieved context
        context_summary = self._format_retrieved_context(retrieved_documents)
        
        # Generate prompt
        prompt = self._build_prompt(cnn_summary, context_summary, patient_info)
        
        # Generate report
        if self.model:
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=GEMINI_CONFIG['temperature'],
                        max_output_tokens=GEMINI_CONFIG['max_output_tokens']
                    )
                )
                return self._format_final_report(response.text)
            except Exception as e:
                print(f"Gemini API error: {e}")
                return self._generate_fallback_report(cnn_results, retrieved_documents, patient_info)
        else:
            return self._generate_fallback_report(cnn_results, retrieved_documents, patient_info)
    
    def _format_cnn_results(self, cnn_results: Dict) -> str:
        """Format CNN analysis results into readable text."""
        return f"""
Classification: {cnn_results.get('classification', 'Unknown')}
Confidence Score: {cnn_results.get('confidence_score', 'N/A')}
Tumor Probability: {cnn_results.get('tumor_probability', 'N/A')}
Normal Probability: {cnn_results.get('normal_probability', 'N/A')}
Analysis Summary: {cnn_results.get('analysis_summary', 'No summary available')}
"""
    
    def _format_retrieved_context(self, documents: List[Dict]) -> str:
        """Format retrieved documents into context string."""
        if not documents:
            return "No relevant medical literature retrieved."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.get('metadata', {}).get('source', 'Unknown Source')
            topic = doc.get('metadata', {}).get('topic', 'General')
            text = doc.get('document', '')
            context_parts.append(f"""
Reference {i} - {topic} (Source: {source}):
{text}
""")
        
        return "\n".join(context_parts)
    
    def _build_prompt(
        self,
        cnn_summary: str,
        context_summary: str,
        patient_info: Optional[Dict]
    ) -> str:
        """Build the complete prompt for report generation."""
        patient_section = ""
        if patient_info:
            patient_section = f"""
## Patient Information (Anonymized):
Age: {patient_info.get('age', 'Not provided')}
Gender: {patient_info.get('gender', 'Not provided')}
Clinical History: {patient_info.get('history', 'Not provided')}
"""
        
        return f"""
You are a medical AI assistant helping to generate diagnostic reports for brain MRI analysis.
Your role is to synthesize CNN image analysis results with relevant medical literature to produce
a comprehensive, professional diagnostic report.

IMPORTANT DISCLAIMERS:
- This is an AI-assisted analysis and should NOT replace professional medical diagnosis
- All findings should be reviewed and confirmed by a qualified radiologist/physician
- This report is for informational purposes only

{patient_section}

## CNN Image Analysis Results:
{cnn_summary}

## Retrieved Medical Literature Context:
{context_summary}

Please generate a comprehensive diagnostic report that includes:

1. **Executive Summary**: Brief overview of findings
2. **Image Analysis Findings**: Detailed interpretation of CNN results
3. **Clinical Context**: Relevant medical literature insights
4. **Risk Assessment**: Based on confidence scores and literature
5. **Recommendations**: Suggested next steps for the patient/physician
6. **Limitations & Disclaimers**: Important caveats about AI-assisted diagnosis

Format the report professionally and ensure medical accuracy based on the provided context.
"""
    
    def _format_final_report(self, raw_report: str) -> str:
        """Format and clean up the generated report."""
        header = """
╔══════════════════════════════════════════════════════════════════════╗
║            RAG-CNN MEDICAL DIAGNOSTIC REPORT                         ║
║            AI-Assisted Brain MRI Analysis                            ║
╚══════════════════════════════════════════════════════════════════════╝
"""
        
        footer = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISCLAIMER: This report is generated by an AI system and is intended 
for informational purposes only. It should NOT be used as a substitute 
for professional medical advice, diagnosis, or treatment. Always consult 
with a qualified healthcare provider for medical decisions.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        return header + raw_report + footer
    
    def _generate_fallback_report(
        self,
        cnn_results: Dict,
        retrieved_documents: List[Dict],
        patient_info: Optional[Dict]
    ) -> str:
        """Generate a template-based report when API is unavailable."""
        
        classification = cnn_results.get('classification', 'Unknown')
        confidence = cnn_results.get('confidence_score', 'N/A')
        tumor_prob = cnn_results.get('tumor_probability', 'N/A')
        summary = cnn_results.get('analysis_summary', 'Analysis pending')
        
        # Get top relevant context
        context_snippets = []
        for doc in retrieved_documents[:3]:
            topic = doc.get('metadata', {}).get('topic', 'General')
            text = doc.get('document', '')[:300]
            context_snippets.append(f"- **{topic}**: {text}...")
        
        context_text = "\n".join(context_snippets) if context_snippets else "No relevant literature retrieved."
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║            RAG-CNN MEDICAL DIAGNOSTIC REPORT                         ║
║            AI-Assisted Brain MRI Analysis                            ║
╚══════════════════════════════════════════════════════════════════════╝

## 1. EXECUTIVE SUMMARY

This report presents the findings from an AI-assisted analysis of brain MRI 
imaging using a Convolutional Neural Network (CNN) integrated with a 
Retrieval-Augmented Generation (RAG) system for medical context.

**Primary Finding**: {classification}
**Analysis Confidence**: {confidence}

---

## 2. IMAGE ANALYSIS FINDINGS

### Classification Results
- **Diagnosis**: {classification}
- **Confidence Score**: {confidence}
- **Tumor Probability**: {tumor_prob}

### Interpretation
{summary}

---

## 3. RELEVANT MEDICAL LITERATURE

The following medical context was retrieved based on the analysis:

{context_text}

---

## 4. RISK ASSESSMENT

Based on the CNN analysis with {confidence} confidence:
{'- Elevated risk indicators detected. Immediate specialist consultation recommended.' if 'Tumor' in classification else '- No significant abnormalities detected. Routine follow-up advised.'}

---

## 5. RECOMMENDATIONS

1. **Immediate**: {'Schedule consultation with a neurologist/neuro-oncologist' if 'Tumor' in classification else 'No immediate action required'}
2. **Short-term**: {'Consider additional imaging (contrast MRI, PET scan)' if 'Tumor' in classification else 'Continue routine health monitoring'}
3. **Follow-up**: Regular monitoring as advised by healthcare provider

---

## 6. LIMITATIONS & DISCLAIMERS

⚠️ **IMPORTANT NOTICES**:

1. This analysis was performed by an AI system and requires validation by 
   qualified medical professionals.

2. The CNN model was trained on a specific dataset and may not generalize 
   to all patient populations.

3. This report does not constitute a medical diagnosis and should not be 
   used as the sole basis for clinical decisions.

4. Patient history, clinical symptoms, and additional tests should be 
   considered for comprehensive evaluation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISCLAIMER: This report is generated by an AI system and is intended 
for informational purposes only. It should NOT be used as a substitute 
for professional medical advice, diagnosis, or treatment. Always consult 
with a qualified healthcare provider for medical decisions.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return report


if __name__ == "__main__":
    # Test the report generator
    generator = MedicalReportGenerator()
    
    # Sample CNN results
    sample_cnn_results = {
        "classification": "Tumor Detected",
        "confidence_score": "87.5%",
        "tumor_probability": "87.5%",
        "normal_probability": "12.5%",
        "analysis_summary": "High-confidence detection of abnormal mass in brain MRI. Immediate specialist consultation recommended."
    }
    
    # Sample retrieved documents
    sample_docs = [
        {
            "document": "Glioblastoma multiforme (GBM) is the most aggressive type of brain cancer. Treatment typically involves surgery, radiation therapy, and chemotherapy.",
            "metadata": {"source": "Oncology Journal", "topic": "Glioblastoma"}
        },
        {
            "document": "MRI is the gold standard for brain tumor detection. T1-weighted images with gadolinium contrast help identify tumor boundaries.",
            "metadata": {"source": "Radiology Handbook", "topic": "MRI Imaging"}
        }
    ]
    
    # Generate report
    report = generator.generate_report(sample_cnn_results, sample_docs)
    print(report)
