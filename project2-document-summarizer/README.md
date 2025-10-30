# Intelligent Document Summarizer with Contextual Q&A# Document Summarization and Q&A System````markdown



A RAG-based system for document analysis using Qwen 1.5B for intelligent summarization and question answering.# Intelligent Document Summarizer with Contextual Q&A



![Screenshot](Screenshot%202025-10-30%20at%2011.41.11.png)A RAG-based system for document analysis using Qwen 1.5B for intelligent summarization and question answering.



## OverviewA powerful offline document analysis system that reads documents (PDFs and text files), generates intelligent summaries, and enables natural language question answering through semantic search.



This project implements a complete document intelligence pipeline for processing PDF and text documents. The system generates intelligent summaries and enables natural language question answering through retrieval-augmented generation (RAG).## Overview



### Key Features##  Overview



- **Document Processing**: PDF and text file support with intelligent chunkingThis system processes PDF and text documents, generates summaries, and enables natural language question answering through retrieval-augmented generation (RAG).

- **Summarization**: LLM-based document summarization using Qwen 1.5B

- **Semantic Search**: FAISS vector store with sentence embeddingsThis project implements a complete document intelligence pipeline using **BART for summarization** and **FAISS for semantic retrieval**. The system:

- **Question Answering**: RAG-based Q&A with context retrieval

- **Offline Operation**: Runs completely locally after initial model download### Key Features

- **Dual Interface**: CLI and web interfaces for flexible usage

- Loads and processes PDF and text documents

## Architecture

- **Document Processing**: PDF and text file support with intelligent chunking- ✂️ Intelligently chunks documents for better processing

### Core Components

- **Summarization**: LLM-based document summarization using Qwen 1.5B- Generates multi-level summaries (chunk + overall)

1. **Document Processor**

   - PDF text extraction using `pypdf`- **Semantic Search**: FAISS vector store with sentence embeddings-  Enables semantic search using sentence embeddings

   - Text file loading

   - LangChain `RecursiveCharacterTextSplitter` for intelligent chunking (1000 chars, 200 overlap)- **Question Answering**: RAG-based Q&A with context retrieval-  Answers natural language questions with context

   - Metadata management

- **Offline Operation**: Runs completely locally after initial model download- 🖥️ Provides both CLI and web interfaces

2. **RAG System** (`DocumentRAGSystem`)

   - Model: `Qwen/Qwen2.5-1.5B-Instruct` for summarization and Q&A-  Runs completely offline (no cloud APIs)

   - Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (90MB)

   - Vector store: FAISS (Facebook AI Similarity Search)## Architecture

   - Document processing pipeline with chunk-based analysis

   - Context-aware question answering## 🏗️ Architecture



3. **Web Interface** (Streamlit)### Components

   - Document upload support (PDF, TXT)

   - Real-time document processing### Core Components

   - Interactive Q&A interface

   - Processing status and summaries display1. **Document Loader**: Extracts text from PDF and TXT files



### Processing Pipeline2. **Text Chunker**: Splits documents into manageable chunks with overlap1. **Document Processor** (`DocumentProcessor`)



```3. **Embedding Generator**: Creates vector embeddings for semantic search   - PDF text extraction using `pypdf`

Document Upload

    ↓4. **Vector Store**: FAISS index for fast similarity search   - Text file loading

Text Extraction (PDF/TXT)

    ↓5. **LLM**: Qwen 1.5B for summarization and answer generation   - LangChain `RecursiveCharacterTextSplitter` for intelligent chunking

Text Chunking (RecursiveCharacterTextSplitter)

    ↓   - Metadata management

Embedding Generation (all-MiniLM-L6-v2)

    ↓### Processing Pipeline

FAISS Index Creation

    ↓2. **Summarization Engine** (`DocumentSummarizer`)

├─→ Summarization (Qwen 1.5B)

│   └─→ Overall summary```   - Model: `facebook/bart-large-cnn` (state-of-the-art summarization)

│

└─→ Question AnsweringDocument → Text Extraction → Chunking → Embedding → FAISS Index   - Chunk-level summarization

    ├─→ Query Embedding

    ├─→ Semantic Retrieval (top-3 chunks)                                              ↓   - Overall document summary from chunk summaries

    └─→ Answer Generation (Qwen 1.5B with context)

```User Question → Query Embedding → Retrieval → LLM Generation → Answer   - Configurable summary lengths



## Requirements```



- Python 3.8+3. **Semantic Search & Q&A** (`SemanticSearchQA`)

- 8GB+ RAM (16GB recommended for large documents)

- ~3GB disk space for models## Requirements   - Embeddings: `sentence-transformers/all-MiniLM-L6-v2`

- macOS, Linux, or Windows

   - Vector store: FAISS (Facebook AI Similarity Search)

## Installation

  - Top-k retrieval for relevant contexts

### 1. Navigate to Project



```bash

cd project2-document-summarizer

```



### 2. Create Virtual Environment

project2-document-summarizer/##  Requirements

```bash

python -m venv venv├── rag_system.py           # Core RAG implementation



# Activate on macOS/Linux:├── streamlit_app.py        # Web interface- Python 3.8+

source venv/bin/activate

├── requirements.txt        # Dependencies- 8GB+ RAM (16GB recommended for large documents)

# Activate on Windows:

venv\Scripts\activate├── test_system.py          # Test suite- ~3GB disk space for models

```

├── data/                   # Sample documents- macOS, Linux, or Windows

### 3. Install Dependencies

│   ├── ai_ml_overview.txt

```bash

pip install -r requirements.txt│   └── climate_change.txt## 🛠️ Installation

```

└── README.md

This will download:

- PyTorch (~2GB)```### 1. Navigate to Project

- Transformers library

- Sentence-Transformers

- FAISS

- Streamlit## Models Used```bash

- pypdf

- LangChain componentscd project2-document-summarizer



**Note**: First run will download Qwen 1.5B model (~3GB) and sentence transformer (~90MB).| Component | Model | Size | Purpose |```



## Usage|-----------|-------|------|---------|



### Option 1: Streamlit Web UI (Recommended)| LLM | Qwen/Qwen2.5-1.5B-Instruct | ~3GB | Summarization & Q&A |### 2. Create Virtual Environment



```bash| Embeddings | all-MiniLM-L6-v2 | 90MB | Semantic search |

streamlit run streamlit_app.py

```| Vector Store | FAISS | - | Similarity search |```bash



Opens at `http://localhost:8501`:````

1. Upload a PDF or text document
2. Wait for processing (shows progress)
3. View automatic summary
4. Ask questions about the document
5. Receive context-aware answers

### Option 2: CLI Mode

```python
from rag_system import DocumentRAGSystem

# Initialize system
rag = DocumentRAGSystem()

# Process a document
rag.process_document("data/ai_ml_overview.txt")

# Get summary
summary = rag.summarize()
print("Summary:", summary)

# Ask questions
answer = rag.answer_question("What are the main topics discussed?")
print("Answer:", answer)
```

## Performance

- **Document Processing**: 5-15 seconds for typical documents
- **Summarization**: 10-30 seconds depending on document length
- **Q&A Response**: 3-8 seconds per question
- **Memory Usage**: ~3-4GB RAM with model loaded
- **Supported File Sizes**: Up to 10MB recommended

## Technology Stack

| Component | Technology | Details |
|-----------|-----------|---------|
| LLM | Qwen/Qwen2.5-1.5B-Instruct | Instruction-tuned, 1.5B parameters, ~3GB |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | 90MB, semantic search |
| Vector Store | FAISS | Facebook AI Similarity Search |
| Text Splitting | LangChain RecursiveCharacterTextSplitter | 1000 char chunks, 200 overlap |
| PDF Processing | pypdf | Text extraction from PDFs |
| Framework | PyTorch 2.0+, Transformers 4.30+ | Model inference |
| UI | Streamlit 1.51.0 | Web interface |

## Project Structure

```
project2-document-summarizer/
├── rag_system.py           # Core RAG implementation
├── streamlit_app.py        # Web interface
├── requirements.txt        # Dependencies
├── data/                   # Sample documents
│   ├── ai_ml_overview.txt
│   └── climate_change.txt
└── README.md
```


Tests include:
- Document processing
- FAISS index creation
- Summarization quality
- Q&A functionality
- Error handling

### Example Test Documents

The `data/` folder includes:
- `ai_ml_overview.txt`: AI/ML concepts overview (~5KB)
- `climate_change.txt`: Climate change information (~3KB)

## Limitations

1. **Document Size**: Large documents (>10MB) may be slow to process
2. **Summary Quality**: Depends on document structure and content clarity
3. **Context Window**: Limited to ~2048 tokens for LLM processing
4. **Language Support**: English only
5. **PDF Limitations**: Complex layouts, tables, or images may not extract properly


## Sample Questions to Try

For `ai_ml_overview.txt`:
- "What is machine learning?"
- "What are the main types of AI?"
- "How does deep learning work?"

For `climate_change.txt`:
- "What are the main causes of climate change?"
- "What are the impacts mentioned?"
- "What solutions are proposed?"

