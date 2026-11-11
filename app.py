"""
Streamlit Web Interface for RAG-CNN Medical Diagnostic System
Provides an interactive UI for MRI analysis
"""
import streamlit as st
import os
from pathlib import Path
from PIL import Image
import tempfile
import PyPDF2
import io

# Set page config first (must be first Streamlit command)
st.set_page_config(
    page_title="RAG-CNN Medical Diagnosis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our modules (after page config)
from main_pipeline import RAGCNNDiagnosticSystem
from rag_module import MedicalVectorDB, SAMPLE_MEDICAL_KNOWLEDGE


@st.cache_resource
def load_system():
    """Load the diagnostic system (cached)."""
    return RAGCNNDiagnosticSystem()


def main():
    # Sidebar
    st.sidebar.title("🧠 RAG-CNN System")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🔬 MRI Analysis", "📚 Knowledge Base", "⚙️ System Status"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **About**
    
    This system combines:
    - CNN (VGG16) for tumor detection
    - FAISS for medical literature retrieval
    - Gemini API for report generation
    """)
    
    # Load system
    with st.spinner("Loading system..."):
        system = load_system()
    
    # Page routing
    if page == "🔬 MRI Analysis":
        show_analysis_page(system)
    elif page == "📚 Knowledge Base":
        show_knowledge_base_page(system)
    else:
        show_status_page(system)


def show_analysis_page(system):
    """MRI Analysis page."""
    st.title("🔬 Brain MRI Analysis")
    st.markdown("Upload a brain MRI image for AI-assisted tumor detection and diagnosis.")
    
    # Check if model is trained
    if not getattr(system.cnn_model, 'is_trained', False):
        st.warning("""
        ⚠️ **Model Not Trained**: The CNN model hasn't been trained yet. Results may be inaccurate.
        
        **To train the model:**
        1. Open `RAG_CNN_Training_Colab.ipynb` in Google Colab
        2. Run all cells to train on brain MRI dataset
        3. Download `brain_tumor_cnn.pth` and place it in the `models/` folder
        4. Restart this app
        """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an MRI image",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload a brain MRI scan image"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI", use_container_width=True)
        
        st.subheader("Patient Info (Optional)")
        patient_age = st.text_input("Age", placeholder="e.g., 45")
        patient_gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
        clinical_history = st.text_area(
            "Clinical History",
            placeholder="e.g., Persistent headaches for 3 months..."
        )
        
        analyze_btn = st.button("🔍 Analyze MRI", type="primary", use_container_width=True)
    
    with col2:
        if analyze_btn and uploaded_file:
            with st.spinner("Analyzing MRI image..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        image.save(tmp.name)
                        tmp_path = tmp.name
                    
                    # Prepare patient info
                    patient_info = None
                    if patient_age or patient_gender or clinical_history:
                        patient_info = {
                            "age": patient_age if patient_age else "Not provided",
                            "gender": patient_gender if patient_gender else "Not provided",
                            "history": clinical_history if clinical_history else "Not provided"
                        }
                    
                    # Run analysis
                    results = system.analyze_mri(
                        tmp_path,
                        patient_info=patient_info,
                        generate_report=True
                    )
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                    
                    # Display results
                    cnn_results = results["cnn_analysis"]
                    
                    # Classification result with color
                    st.subheader("Classification Result")
                    if "Tumor" in cnn_results["classification"]:
                        st.error(f"⚠️ **{cnn_results['classification']}**")
                    else:
                        st.success(f"✅ **{cnn_results['classification']}**")
                    
                    # Metrics
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Confidence", cnn_results["confidence_score"])
                    with col_b:
                        st.metric("Tumor Probability", cnn_results["tumor_probability"])
                    
                    # Analysis summary
                    st.subheader("Analysis Summary")
                    st.info(cnn_results["analysis_summary"])
                    
                    # Retrieved documents
                    st.subheader("Retrieved Medical Context")
                    for i, doc in enumerate(results["retrieved_documents"][:3], 1):
                        topic = doc.get("metadata", {}).get("topic", "General")
                        with st.expander(f"📄 {topic}"):
                            st.write(doc["document"])
                    
                    # Full report
                    st.subheader("Detailed Diagnostic Report")
                    with st.expander("📋 View Full Report", expanded=True):
                        st.markdown(results["report"])
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
        
        elif analyze_btn and not uploaded_file:
            st.warning("Please upload an MRI image first.")
        
        else:
            st.info("👆 Upload an MRI image and click 'Analyze MRI' to get started.")


def show_knowledge_base_page(system):
    """Knowledge Base Chatbot page."""
    st.title("📚 Medical Knowledge Chatbot")
    st.markdown("Chat with your medical knowledge base. Upload PDFs and ask questions!")
    
    # Initialize chat history
    if "kb_messages" not in st.session_state:
        st.session_state.kb_messages = []
    
    # Sidebar for document upload and MRI analysis
    with st.sidebar:
        st.markdown("---")
        st.subheader("🧠 Analyze MRI")
        
        # MRI Upload
        mri_image = st.file_uploader(
            "Upload MRI Scan",
            type=["png", "jpg", "jpeg"],
            help="Upload brain MRI image for tumor detection",
            key="kb_mri_upload"
        )
        
        if mri_image:
            st.image(mri_image, caption="MRI Preview", use_container_width=True)
            if st.button("🔬 Analyze MRI", key="kb_analyze_mri"):
                with st.spinner("Analyzing MRI..."):
                    try:
                        # Analyze the image
                        image = Image.open(mri_image)
                        results = system.analyze_mri(image)
                        
                        # Store results in session state for chat context
                        st.session_state.last_mri_result = results
                        
                        # Extract classification info
                        classification = results['summary']['classification']
                        confidence = results['summary']['confidence']
                        tumor_detected = results['summary']['tumor_detected']
                        
                        # Create chat message with results
                        mri_msg = f"""🧠 **MRI Analysis Complete!**

**Classification:** {classification}
**Confidence:** {confidence}

"""
                        if tumor_detected:
                            mri_msg += "⚠️ The CNN model detected potential tumor indicators in this scan.\n\n"
                        else:
                            mri_msg += "✅ No tumor indicators detected in this scan.\n\n"
                        
                        mri_msg += "*Would you like to know more about brain tumors or ask questions about this result?*"
                        
                        st.session_state.kb_messages.append({
                            "role": "assistant",
                            "content": mri_msg
                        })
                        st.success("Analysis added to chat!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        st.markdown("---")
        st.subheader("📄 Upload Documents")
        
        # PDF Upload
        uploaded_pdf = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            help="Upload medical research papers or documents",
            key="kb_pdf_upload"
        )
        
        if uploaded_pdf:
            pdf_topic = st.text_input("Topic (optional)", key="kb_pdf_topic", placeholder="e.g., Brain Tumor")
            
            if st.button("📤 Add to Knowledge Base", key="kb_add_pdf"):
                with st.spinner("Processing PDF..."):
                    try:
                        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_pdf.read()))
                        full_text = ""
                        for page in pdf_reader.pages:
                            text = page.extract_text()
                            if text:
                                full_text += text + "\n"
                        
                        if full_text.strip():
                            # Split into chunks
                            chunks = []
                            words = full_text.split()
                            current_chunk = ""
                            chunk_size = 1000
                            
                            for word in words:
                                if len(current_chunk) + len(word) + 1 <= chunk_size:
                                    current_chunk += word + " "
                                else:
                                    if current_chunk.strip():
                                        chunks.append(current_chunk.strip())
                                    current_chunk = word + " "
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                            
                            metadata_list = [{
                                "topic": pdf_topic if pdf_topic else uploaded_pdf.name,
                                "source": "PDF Upload"
                            } for _ in chunks]
                            
                            system.add_medical_documents(chunks, metadata_list)
                            
                            st.success(f"✅ Added {len(chunks)} chunks!")
                            
                            # Add system message to chat
                            st.session_state.kb_messages.append({
                                "role": "system",
                                "content": f"📄 **Document Added**: {uploaded_pdf.name}\n- {len(chunks)} text chunks extracted\n- Topic: {pdf_topic if pdf_topic else 'General'}"
                            })
                            st.rerun()
                        else:
                            st.error("Could not extract text from PDF")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        st.markdown("---")
        st.markdown(f"**📊 Documents:** {system.vector_db.get_stats()['total_documents']}")
        
        if st.button("🗑️ Clear Chat", key="kb_clear"):
            st.session_state.kb_messages = []
            st.rerun()
    
    # Welcome message if no messages
    if not st.session_state.kb_messages:
        st.session_state.kb_messages.append({
            "role": "assistant",
            "content": """👋 **Hello! I'm your Medical Knowledge Assistant.**

I can help you with questions about medical topics and analyze brain MRI scans!

**Here's what you can do:**
- 🧠 **Upload MRI scans** in the sidebar for brain tumor detection
- 📄 **Upload PDFs** to expand my knowledge base
- 💬 **Ask me questions** about medical topics or your MRI results
- 🔍 I'll combine CNN analysis with my medical knowledge

**Try asking me something like:**
- "What are the symptoms of brain tumors?"
- "Can you explain my MRI results?"
- "What does glioblastoma look like on MRI?"

*Note: I analyze MRIs using a trained CNN model and answer questions from my knowledge base. Upload documents to expand what I know!*"""
        })
    
    # Chat container
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.kb_messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(message["content"])
            elif message["role"] == "system":
                with st.chat_message("assistant", avatar="📄"):
                    st.info(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about medical topics..."):
        # Add user message
        st.session_state.kb_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Search knowledge base
                results = system.search_knowledge_base(prompt, top_k=5)
                
                # Build conversation history for context
                chat_history = ""
                for msg in st.session_state.kb_messages[-6:]:  # Last 3 exchanges
                    if msg["role"] == "user":
                        chat_history += f"User: {msg['content']}\n"
                    elif msg["role"] == "assistant" and "Hello! I'm your" not in msg["content"]:
                        chat_history += f"Assistant: {msg['content'][:200]}...\n"
                
                # Check if there's recent MRI analysis
                mri_context = ""
                if "last_mri_result" in st.session_state:
                    mri_result = st.session_state.last_mri_result
                    classification = mri_result['summary']['classification']
                    confidence = mri_result['summary']['confidence']
                    tumor_detected = mri_result['summary']['tumor_detected']
                    mri_context = f"""
Recent MRI Analysis Result:
- Classification: {classification}
- Confidence: {confidence}
- Tumor Detected: {'Yes' if tumor_detected else 'No'}
- The user may ask questions about this MRI scan result.
"""
                
                if results or mri_context:
                    # Build context from results
                    context = "\n\n".join([r['document'] for r in results[:4]]) if results else ""
                    sources = [f"{r.get('metadata', {}).get('topic', 'Document')}" for r in results[:3]] if results else []
                    
                    # Generate conversational response using Gemini
                    if system.report_generator.model:
                        try:
                            gen_prompt = f"""You are a friendly and helpful medical knowledge assistant with access to a Brain Tumor CNN model. You can answer questions based on the provided knowledge base documents AND recent MRI analysis results. Be conversational, warm, and helpful.

IMPORTANT RULES:
1. Use information from the "Knowledge Base Context" and "MRI Analysis" sections below
2. If the user asks about their MRI result, explain it clearly using the MRI Analysis data
3. If the question cannot be answered from the context, politely say you don't have that information
4. Be conversational and natural, like chatting with a knowledgeable doctor
5. Use simple language and explain medical terms when needed
6. If the MRI shows tumor detected, be empathetic but informative - recommend consulting a doctor
7. If appropriate, suggest follow-up questions the user might want to ask

Previous conversation:
{chat_history}
{mri_context}
Knowledge Base Context:
{context if context else "No specific documents found for this query."}

User's Question: {prompt}

Respond naturally and helpfully. If discussing MRI results, explain what the CNN model detected and what it might mean. Always recommend professional medical consultation for any concerning findings."""

                            response = system.report_generator.model.generate_content(gen_prompt)
                            answer = response.text
                            
                            # Add subtle source reference
                            if len(sources) > 0:
                                unique_sources = list(set(sources))[:3]
                                answer += f"\n\n---\n📚 *Based on: {', '.join(unique_sources)}*"
                            elif mri_context:
                                answer += f"\n\n---\n🧠 *Based on CNN MRI Analysis*"
                        except Exception as e:
                            # Fallback response
                            answer = f"Based on my knowledge base, here's what I found:\n\n"
                            if mri_context:
                                mri_r = st.session_state.last_mri_result
                                answer += f"**Your MRI Analysis:** {mri_r['summary']['classification']} ({mri_r['summary']['confidence']} confidence)\n\n"
                            for i, result in enumerate(results[:2], 1):
                                topic = result.get("metadata", {}).get("topic", "General")
                                answer += f"**{topic}:** {result['document'][:400]}...\n\n"
                    else:
                        # No Gemini - show raw results conversationally
                        answer = "Let me share what I found:\n\n"
                        if mri_context:
                            mri_r = st.session_state.last_mri_result
                            answer += f"🧠 **Your MRI Analysis:** {mri_r['summary']['classification']} ({mri_r['summary']['confidence']} confidence)\n\n"
                        for i, result in enumerate(results[:2], 1):
                            topic = result.get("metadata", {}).get("topic", "General")
                            answer += f"📖 **{topic}:**\n{result['document'][:400]}...\n\n"
                else:
                    # No results found - be helpful
                    answer = """I'm sorry, I couldn't find information about that in my current knowledge base. 😔

**Here's what you can do:**
- 🧠 Upload an MRI scan in the sidebar for tumor detection
- 📄 Upload relevant PDF documents to expand my knowledge
- 🔄 Try rephrasing your question
- ❓ Ask about topics related to brain tumors, MRI imaging, or medical diagnosis

Is there something else I can help you with?"""
                
                st.markdown(answer)
        
        # Add assistant message to history
        st.session_state.kb_messages.append({"role": "assistant", "content": answer})


def show_status_page(system):
    """System Status page."""
    st.title("⚙️ System Status")
    
    status = system.get_system_status()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🧠 CNN Model")
        st.metric("Architecture", status["cnn_model"]["architecture"])
        device_str = str(status["cnn_model"]["device"])
        if "cuda" in device_str.lower():
            st.success(f"🚀 GPU: {device_str}")
        else:
            st.warning(f"💻 {device_str}")
        st.write("**Classes:**", ", ".join(status["cnn_model"]["classes"]))
        
        # Check if model is trained
        if status["cnn_model"].get("is_trained", False):
            st.success("✅ Model Trained")
        else:
            st.error("❌ Model NOT Trained")
            st.warning("Upload `brain_tumor_cnn.pth` to `models/` folder after training in Colab")
    
    with col2:
        st.subheader("📚 Vector Database")
        st.metric("Total Documents", status["vector_database"]["total_documents"])
        st.metric("Embedding Dim", status["vector_database"]["embedding_dimension"])
        st.write("**Model:**", status["vector_database"]["embedding_model"])
    
    with col3:
        st.subheader("📝 Report Generator")
        st.metric("LLM Model", status["report_generator"]["model"])
        if status["report_generator"]["api_configured"]:
            st.success("✅ API Configured")
        else:
            st.warning("⚠️ Using Fallback (Add GOOGLE_API_KEY to .env)")
    
    st.markdown("---")
    
    # Knowledge base preview
    st.subheader("📖 Knowledge Base Preview")
    
    if st.button("🔄 Refresh"):
        st.rerun()
    
    docs = system.vector_db.documents[:5]
    for i, doc in enumerate(docs, 1):
        metadata = system.vector_db.metadata[i-1] if i-1 < len(system.vector_db.metadata) else {}
        topic = metadata.get("topic", "General")
        with st.expander(f"{i}. {topic}"):
            st.write(doc[:300] + "..." if len(doc) > 300 else doc)


# Footer
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 0.9em;">
        ⚠️ <strong>Disclaimer:</strong> This is an AI-assisted diagnostic tool for educational purposes only. 
        It should NOT be used as a substitute for professional medical advice.<br>
        <em>RAG-CNN Framework | UROP Project 2025</em>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    show_footer()
