# Tech Support Chatbot

A domain-specific conversational AI chatbot for technical support, built with local LLMs and running entirely offline.

![Screenshot](Screenshot%202025-10-30%20at%2011.40.18.png)

## Overview

This project implements an intelligent tech support chatbot that can:
- Handle multi-turn conversations with context awareness
- Retrieve relevant information from a local knowledge base using RAG (Retrieval-Augmented Generation)
- Provide accurate answers to technical support questions
- Run completely offline using local models (no cloud APIs)

The chatbot is designed for **TechSoft Pro Suite** technical support, covering topics like account management, billing, troubleshooting, features, and security.

## Architecture

### Core Components

1. **LLM Engine** (`Qwen/Qwen2.5-1.5B-Instruct`)
   - Instruction-tuned model for better conversational quality
   - Runs locally via Hugging Face Transformers
   - 1.5B parameters, optimized for dialogue and Q&A
   - Generates responses with retrieved context

2. **Semantic Search System**
   - Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
   - Vector store: FAISS for similarity search
   - LangChain `RecursiveCharacterTextSplitter` for intelligent document chunking
   - Retrieves top-2 relevant contexts per query

3. **Conversation Management**
   - Manual conversation history tracking (last 5 exchanges)
   - Context-aware response generation
   - Fallback mechanism for failed generations

4. **Knowledge Base**
   - `data/tech_support_faq.csv`: 20 common support Q&As
   - `data/product_info.txt`: Comprehensive product documentation
   - Covers pricing, features, integrations, security, and more

5. **User Interfaces**
   - **CLI Mode** (`chatbot.py`): Terminal-based interaction
   - **Web UI** (`app.py`): Streamlit-based graphical interface

### Data Flow

```
User Query
    ↓
FAISS Semantic Search
    ↓
Retrieve Relevant Context (top-2 chunks)
    ↓
Combine with Conversation History
    ↓
Generate Prompt for Qwen 1.5B
    ↓
LLM Response Generation
    ↓
Response Extraction & Fallback
    ↓
Update Conversation History
```

### Key Design Decisions

- **Direct RAG Implementation** for simplicity and control
- **Manual Memory Management** with last 5 conversation turns
- **Document Chunking** with LangChain's `RecursiveCharacterTextSplitter` for better context retrieval
- **Fallback mechanism** when LLM output is poor or empty
- **Modular design** separating chatbot logic from UI
- **Qwen 1.5B** for superior instruction-following and Q&A quality

## Features

- **Multi-turn Dialogue**: Maintains context across conversation exchanges
- **Contextual Retrieval**: Semantic search via FAISS to find relevant information
- **Offline Operation**: No internet required after initial setup
- **Two Interfaces**: CLI for testing, Streamlit for user-friendly interaction
- **Conversation Memory**: Tracks last 5 conversation turns
- **Smart Fallbacks**: Handles edge cases when generation fails
- **Qwen 1.5B**: Instruction-tuned LLM for high-quality responses

## Requirements

- Python 3.8+
- 4GB+ RAM (8GB recommended)
- ~3GB disk space for models
- macOS, Linux, or Windows

## Installation

### 1. Clone/Navigate to Project

```bash
cd project1-chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Activate on macOS/Linux:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will download:
- Transformers and PyTorch (~2GB)
- Sentence-Transformers
- FAISS
- Streamlit
- Accelerate
- Other utilities

**Note**: First-time model downloads may take 5-10 minutes depending on your connection.

## Usage

### Option 1: CLI Mode

```bash
python chatbot.py
```

Simple terminal interface for quick testing:

```
You: How do I reset my password?
Bot: To reset your password: 1) Click 'Forgot Password' on the login page...

You: What about 2FA?
Bot: You can enable two-factor authentication in Settings > Security...
```

Commands:
- Type your question and press Enter
- `reset` - Clear conversation history
- `quit` or `exit` - Exit the chatbot

### Option 2: Streamlit Web UI (Recommended)

```bash
streamlit run app.py
```

This opens a web interface at `http://localhost:8501` with:
- Clean chat interface
- Conversation history
- Example questions
- Statistics sidebar
- Clear conversation button

## Performance

- **Initial Load**: ~10-30 seconds (model loading)
- **Response Time**: 2-5 seconds per query
- **Memory Usage**: ~2-3GB RAM
- **Accuracy**: Good for in-domain queries, limited for out-of-scope questions

## Limitations

### Model Limitations
1. **Domain-Specific**: Trained on TechSoft Pro Suite data only
   - May not handle general queries well
   - Best for tech support scenarios covered in knowledge base

2. **Generation Quality**: 1.5B parameter model
   - Generally produces good responses for in-domain queries
   - Fallback mechanism mitigates edge cases by using direct context

3. **Context Window**: Limited to ~512 tokens
   - Very long conversations may lose early context
   - History limited to last 5 exchanges

### Technical Limitations
4. **No Real-Time Learning**: Cannot learn from new conversations
   - Knowledge base is static
   - Would need retraining or manual KB updates

5. **Computational Requirements**
   - Requires decent hardware (4GB+ RAM)
   - CPU inference is slower than GPU
   - Initial load time on first run

6. **Language Support**: English only
   - Models not trained for multilingual support

### Potential Improvements
- Use larger/better models (GPT-J, Llama 2, Mistral)
- Implement GPU acceleration for faster inference
- Add conversation summarization for longer contexts
- Dynamic knowledge base updates
- Fine-tuning on domain-specific dialogues
- Add confidence scoring and "I don't know" responses
- Implement streaming responses for better UX

## Technology Stack

| Component | Technology | Details |
|-----------|-----------|---------|
| LLM | Qwen/Qwen2.5-1.5B-Instruct | Instruction-tuned, 1.5B parameters, ~3GB |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | 90MB, semantic search |
| Vector Store | FAISS | Facebook AI Similarity Search |
| Text Splitting | LangChain RecursiveCharacterTextSplitter | 1000 char chunks, 200 overlap |
| Framework | PyTorch 2.0+, Transformers 4.30+ | Model inference |
| UI | Streamlit 1.51.0 | Web interface |
| Accelerator | accelerate 1.11.0 | Efficient model loading |

## Project Structure

```
project1-chatbot/
├── chatbot.py              # Core chatbot logic and CLI
├── app.py                  # Streamlit web interface
├── requirements.txt        # Python dependencies
├── data/
│   ├── tech_support_faq.csv    # FAQ questions and answers
│   └── product_info.txt        # Product documentation
├── README.md              # This file
```

## Testing

### Basic Functionality Test

```python
from chatbot import TechSupportChatbot

bot = TechSupportChatbot()

# Single query
response = bot.chat("How do I reset my password?")
print(response)

# Multi-turn
bot.chat("What's the Pro plan price?")
response = bot.chat("What features does it include?")
print(response)

# Check history
print(bot.get_conversation_history())
```

### Example Questions to Try

- "How do I reset my password?"
- "What's included in the Pro plan?"
- "Is my data secure?"
- "How do I cancel my subscription?"
- "What file formats are supported?"
- "Can I use it on mobile?"

## Customization

### Change the Domain

1. Update `data/tech_support_faq.csv` with your FAQs
2. Update `data/product_info.txt` with your documentation
3. Optionally change model in `chatbot.py`:
   ```python
   chatbot = TechSupportChatbot(model_name="Qwen/Qwen2.5-3B-Instruct")
   ```

### Adjust Parameters

In `chatbot.py`, modify:
- `max_history`: Number of conversation exchanges to remember
- `top_k`: Number of context chunks to retrieve
- Generation parameters: `temperature`, `top_p`, `max_new_tokens`

