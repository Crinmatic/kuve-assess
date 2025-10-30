# AI Engineer Practical Assessment - Kuve

This repository contains three complete AI/ML projects demonstrating practical skills in natural language processing, deep learning, and production deployment.

## Projects Overview

### Project 1: Domain-Specific Chatbot
**Tech Support Chatbot with RAG**

A conversational AI system for TechSoft Pro Suite technical support using retrieval-augmented generation.

- **Technology**: Qwen 1.5B, FAISS, sentence-transformers
- **Features**: Multi-turn dialogue, semantic search, conversation memory
- **Interface**: Streamlit web UI + CLI
- **Capabilities**: Handles technical support queries with context-aware responses

📁 [View Project 1](./project1-chatbot/)

---

### Project 2: Intelligent Document Summarizer
**Document Analysis with Contextual Q&A**

A RAG-based system for processing documents, generating summaries, and answering questions.

- **Technology**: Qwen 1.5B, FAISS, pypdf, LangChain
- **Features**: PDF/text processing, intelligent summarization, semantic Q&A
- **Interface**: Streamlit web UI + Python API
- **Capabilities**: Document understanding, context retrieval, natural language queries

📁 [View Project 2](./project2-document-summarizer/)

---

### Project 3: Text Classification Model
**Sentiment Analysis with Production API**

Binary sentiment classification using fine-tuned DistilBERT with FastAPI deployment.

- **Technology**: DistilBERT, PyTorch, FastAPI, Transformers
- **Features**: 92-94% accuracy, comprehensive evaluation, production-ready API
- **Interface**: REST API with interactive docs
- **Capabilities**: Single/batch prediction, real-time inference, error handling

📁 [View Project 3](./project3-classifier/)

---

## Technology Stack

| Component | Technologies |
|-----------|-------------|
| **LLMs** | Qwen 2.5-1.5B-Instruct, DistilBERT |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **Frameworks** | PyTorch, Transformers, LangChain |
| **APIs** | FastAPI, Uvicorn |
| **UI** | Streamlit |
| **Tools** | pandas, scikit-learn, nltk |

## Key Achievements

**End-to-End ML Pipeline**: Data processing, model training, evaluation, deployment  
**Production-Ready Code**: Clean architecture, error handling, documentation  
**Offline Operation**: All systems run locally without cloud dependencies  
**Comprehensive Evaluation**: Multiple metrics, visualizations, error analysis  
**Interactive Interfaces**: Web UIs and REST APIs for easy demonstration  
**Best Practices**: Virtual environments, requirements files, testing suites  

## Quick Start

Each project includes:
- Complete README with installation instructions
- Requirements file for dependencies
- Demo scripts or notebooks
- Screenshots of working systems

### Running a Project

```bash
# Example: Project 1
cd project1-chatbot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
kuve/
├── README.md                          # This file
├── INTERVIEW_QUESTIONS.md             # Assessment requirements
├── project1-chatbot/                  # RAG-based tech support chatbot
│   ├── chatbot.py                     # Core logic
│   ├── app.py                         # Streamlit UI
│   ├── data/                          # Knowledge base
│   └── README.md
├── project2-document-summarizer/      # Document analysis system
│   ├── rag_system.py                  # RAG implementation
│   ├── streamlit_app.py               # Web interface
│   ├── data/                          # Sample documents
│   └── README.md
└── project3-classifier/               # Sentiment classification
    ├── train_sentiment_model.ipynb    # Training notebook
    ├── api.py                         # FastAPI server
    ├── model/                         # Saved model
    └── README.md
```


