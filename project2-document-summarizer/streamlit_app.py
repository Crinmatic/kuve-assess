"""
Streamlit Web Interface for Document RAG System
Professional UI for assessment
"""

import streamlit as st
import os
from pathlib import Path
import tempfile

from rag_system import DocumentRAGSystem


st.set_page_config(
    page_title="Document RAG System",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.document_processed = False
    st.session_state.qa_history = []
    st.session_state.current_file = None
    st.session_state.summary = None


def initialize_system():
    """Initialize the RAG system."""
    with st.spinner("Initializing system..."):
        st.session_state.rag_system = DocumentRAGSystem()
        st.session_state.document_processed = False
        st.session_state.qa_history = []


def process_uploaded_file(uploaded_file):
    """Process an uploaded file."""
    suffix = Path(uploaded_file.name).suffix
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name
    
    try:
        with st.spinner("Processing document..."):
            result = st.session_state.rag_system.process_document(tmp_path)
        
        st.session_state.document_processed = True
        st.session_state.current_file = uploaded_file.name
        st.session_state.summary = result['summary']
        st.session_state.doc_info = result
        st.session_state.qa_history = []
        
        return True
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    st.title("Document Summarization and Q&A System")
    st.markdown("RAG-based system using Qwen 1.5B")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Initialize
        if st.session_state.rag_system is None:
            if st.button("Initialize System", type="primary"):
                initialize_system()
                st.success("System initialized")
                st.rerun()
        
        st.divider()
        
        # Upload
        st.header("Document Upload")
        
        if st.session_state.rag_system is not None:
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['pdf', 'txt'],
                help="Upload PDF or text file"
            )
            
            if uploaded_file:
                if st.button("Process Document", type="primary"):
                    if process_uploaded_file(uploaded_file):
                        st.success("Document processed")
                        st.rerun()
        else:
            st.warning("Initialize system first")
        
        st.divider()
        
        # Status
        st.header("Status")
        
        if st.session_state.rag_system:
            st.success("System ready")
        else:
            st.warning("System not initialized")
        
        if st.session_state.document_processed:
            st.success("Document loaded")
            st.info(f"File: {st.session_state.current_file}")
            
            if 'doc_info' in st.session_state:
                info = st.session_state.doc_info
                st.metric("Chunks", info['num_chunks'])
                st.metric("Characters", f"{info['text_length']:,}")
        else:
            st.warning("No document loaded")
        
        st.divider()
        
        # Settings
        st.header("Settings")
        num_contexts = st.slider(
            "Context chunks for Q&A",
            min_value=1,
            max_value=5,
            value=3
        )
        
        if st.session_state.document_processed:
            if st.button("Clear Document"):
                st.session_state.document_processed = False
                st.session_state.qa_history = []
                st.session_state.current_file = None
                st.session_state.summary = None
                st.rerun()
    
    # Main content
    if st.session_state.rag_system is None:
        st.info("Initialize the system using the sidebar to get started")
        
        st.subheader("Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Document Processing**
            - PDF and text file support
            - Automatic text extraction
            - Intelligent chunking
            
            **Summarization**
            - LLM-based summarization
            - Context-aware summaries
            - Qwen 1.5B model
            """)
        
        with col2:
            st.markdown("""
            **Retrieval (RAG)**
            - Semantic search with embeddings
            - FAISS vector store
            - Fast similarity search
            
            **Question Answering**
            - Natural language generation
            - Context-based answers
            - Source attribution
            """)
        
    elif not st.session_state.document_processed:
        st.info("Upload a document using the sidebar")
        
        st.markdown("""
        **How to use:**
        1. Upload a PDF or text document
        2. Click "Process Document"
        3. View the generated summary
        4. Ask questions about the content
        """)
        
    else:
        # Summary section
        st.header("Document Summary")
        
        with st.expander("View Summary", expanded=True):
            st.markdown(st.session_state.summary)
        
        st.divider()
        
        # Q&A section
        st.header("Question & Answer")
        
        question = st.text_input(
            "Your question:",
            placeholder="Ask anything about the document...",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            ask_button = st.button("Ask", type="primary")
        with col2:
            if st.button("Clear History"):
                st.session_state.qa_history = []
                st.rerun()
        
        if ask_button and question:
            with st.spinner("Generating answer..."):
                answer_data = st.session_state.rag_system.answer_question(
                    question, 
                    k=num_contexts
                )
            
            st.session_state.qa_history.append({
                'question': question,
                'answer': answer_data
            })
            
            st.rerun()
        
        # History
        if st.session_state.qa_history:
            st.divider()
            st.subheader("Conversation History")
            
            for i, qa in enumerate(reversed(st.session_state.qa_history)):
                with st.container():
                    st.markdown(f"**Q{len(st.session_state.qa_history) - i}:** {qa['question']}")
                    st.markdown(f"**A:** {qa['answer']['answer']}")
                    
                    with st.expander("View Sources"):
                        for idx, source in enumerate(qa['answer'].get('sources', []), 1):
                            st.markdown(f"**Source {idx}:**")
                            st.text(source[:300] + "..." if len(source) > 300 else source)
                    
                    st.divider()


if __name__ == "__main__":
    main()
