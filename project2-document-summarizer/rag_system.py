"""
Document Summarization and Q&A System using RAG
"""

import os
from typing import List, Dict, Optional
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentRAGSystem:
    """Complete RAG system for document summarization and Q&A."""
    
    def __init__(
        self, 
        llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """Initialize the RAG system."""
        print(f"Loading models...")
        
        # LLM for generation (summarization + Q&A)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        # Embedding model for retrieval
        self.embedding_model = SentenceTransformer(embedding_model)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Storage
        self.index = None
        self.documents = []
        self.current_file = None
        
        print("System ready")
    
    def load_document(self, file_path: str) -> str:
        """Load document from PDF or TXT file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if path.suffix.lower() == '.pdf':
            reader = PdfReader(file_path)
            text = "\n".join(page.extract_text() for page in reader.pages)
            return text.strip()
        elif path.suffix.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
    
    def chunk_text(self, text: str) -> List[Document]:
        """Split text into chunks."""
        chunks = self.text_splitter.split_text(text)
        return [Document(page_content=chunk, metadata={"chunk_id": i}) 
                for i, chunk in enumerate(chunks)]
    
    def build_index(self, documents: List[Document]):
        """Build FAISS index for retrieval."""
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        
        print(f"Building index for {len(documents)} chunks...")
        self.embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.embeddings.astype('float32'))
        print("Index ready")
    
    def retrieve_context(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant chunks."""
        if self.index is None:
            return []
        
        query_vector = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vector.astype('float32'), k)
        
        contexts = []
        for idx in indices[0]:
            if idx < len(self.documents):
                contexts.append(self.documents[idx].page_content)
        
        return contexts
    
    def generate_text(self, prompt: str, system_msg: str, max_tokens: int = 512) -> str:
        """Generate text using LLM."""
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return generated.strip()
    
    def summarize(self, text: str = None) -> str:
        """Generate document summary."""
        if text is None:
            # Summarize loaded document
            if not self.documents:
                return "No document loaded"
            text = "\n\n".join(doc.page_content for doc in self.documents[:5])  # First 5 chunks
        
        prompt = f"Summarize the following document concisely:\n\n{text[:4000]}"  # Limit context
        system_msg = "You are a helpful assistant that creates concise, accurate summaries."
        
        print("Generating summary...")
        return self.generate_text(prompt, system_msg, max_tokens=300)
    
    def answer_question(self, question: str, k: int = 3) -> Dict:
        """Answer question using RAG."""
        if not self.documents:
            return {"answer": "No document loaded", "sources": []}
        
        # Retrieve context
        contexts = self.retrieve_context(question, k=k)
        
        if not contexts:
            return {"answer": "Could not find relevant information", "sources": []}
        
        # Generate answer
        context_text = "\n\n".join(contexts)
        prompt = f"""Based on the following context, answer the question concisely and accurately.

Context:
{context_text}

Question: {question}

Answer:"""
        
        system_msg = "You are a helpful assistant that answers questions based on provided context. Be concise and factual."
        
        print("Generating answer...")
        answer = self.generate_text(prompt, system_msg, max_tokens=256)
        
        return {
            "answer": answer,
            "sources": contexts
        }
    
    def process_document(self, file_path: str) -> Dict:
        """Load and process a document."""
        print(f"Processing: {file_path}")
        
        # Load
        text = self.load_document(file_path)
        print(f"Loaded {len(text)} characters")
        
        # Chunk
        self.documents = self.chunk_text(text)
        print(f"Created {len(self.documents)} chunks")
        
        # Index
        self.build_index(self.documents)
        
        # Summarize
        summary = self.summarize()
        
        self.current_file = file_path
        
        return {
            "file_path": file_path,
            "num_chunks": len(self.documents),
            "summary": summary,
            "text_length": len(text)
        }


def main():
    """CLI interface."""
    print("=" * 60)
    print("Document RAG System - CLI")
    print("=" * 60)
    
    system = DocumentRAGSystem()
    
    file_path = input("\nEnter document path: ").strip()
    
    if not file_path:
        print("No file path provided")
        return
    
    # Process
    result = system.process_document(file_path)
    
    print("\nSUMMARY:")
    print("-" * 60)
    print(result['summary'])
    print("-" * 60)
    
    # Q&A loop
    print("\nAsk questions (type 'quit' to exit):\n")
    
    while True:
        question = input("Question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        answer = system.answer_question(question)
        print(f"\nAnswer: {answer['answer']}\n")


if __name__ == "__main__":
    main()
