"""
Domain-Specific Tech Support Chatbot
Simplified RAG-based chatbot using Qwen
"""

import os
import pandas as pd
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class TechSupportChatbot:
    """Tech support chatbot with RAG using Qwen 1.5B."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        """Initialize chatbot with Qwen model."""
        self.model_name = model_name
        
        print("Initializing Tech Support Chatbot...")
        print(f"Loading model: {model_name}")
        
        # Load Qwen model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        # Load embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Load knowledge base
        print("Loading knowledge base...")
        self.documents = self._load_knowledge()
        
        # Build FAISS index
        print("Building vector store...")
        texts = [doc['text'] for doc in self.documents]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Conversation history
        self.history = []
        
        print("Chatbot ready!\n")
    
    def _load_knowledge(self) -> List[Dict]:
        """Load knowledge base from files."""
        documents = []
        
        # Load FAQ
        faq_path = os.path.join("data", "tech_support_faq.csv")
        if os.path.exists(faq_path):
            faq_data = pd.read_csv(faq_path)
            for _, row in faq_data.iterrows():
                documents.append({
                    'text': f"Q: {row['question']}\nA: {row['answer']}",
                    'source': 'faq'
                })
        
        # Load product info
        kb_path = os.path.join("data", "product_info.txt")
        if os.path.exists(kb_path):
            with open(kb_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into chunks
            chunks = content.split('\n\n')
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    documents.append({
                        'text': chunk.strip(),
                        'source': f'product_info_{i}'
                    })
        
        return documents
    
    def _retrieve_context(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant contexts using FAISS."""
        query_vector = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vector.astype('float32'), k)
        
        contexts = []
        for idx in indices[0]:
            if idx < len(self.documents):
                contexts.append(self.documents[idx]['text'])
        
        return contexts
    
    def _generate_response(self, question: str, contexts: List[str]) -> str:
        """Generate response using Qwen."""
        context_text = "\n\n".join(contexts[:2])
        
        # Add recent history for context
        history_text = ""
        if self.history:
            recent = self.history[-2:]  # Last 2 exchanges
            history_text = "\n".join([f"User: {h['user']}\nAssistant: {h['bot']}" for h in recent])
        
        prompt = f"""You are a helpful tech support assistant for TechSoft Pro Suite.
Use the context below to answer the question. Be concise and helpful.

Context:
{context_text}

{f'Previous conversation:{history_text}' if history_text else ''}

Question: {question}

Answer:"""
        
        messages = [
            {"role": "system", "content": "You are a helpful tech support assistant. Be concise and clear."},
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
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def chat(self, user_input: str) -> str:
        """Main chat interface."""
        if not user_input.strip():
            return "Please enter a message."
        
        try:
            # Retrieve context
            contexts = self._retrieve_context(user_input)
            
            # Generate response
            response = self._generate_response(user_input, contexts)
            
            # Update history
            self.history.append({'user': user_input, 'bot': response})
            if len(self.history) > 5:
                self.history = self.history[-5:]
            
            return response
        
        except Exception as e:
            print(f"Error: {e}")
            # Fallback to direct context
            if contexts:
                return f"Based on our documentation: {contexts[0][:300]}..."
            return "I apologize, but I encountered an error. Please contact support@techsoft.com."
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.history = []
        print("Conversation history cleared.")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.history


def main():
    """CLI interface."""
    print("=" * 60)
    print("Tech Support Chatbot - CLI Mode")
    print("=" * 60)
    
    chatbot = TechSupportChatbot()
    
    print("\nStart chatting! (Type 'quit' to exit, 'reset' to clear history)\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if user_input.lower() == 'reset':
            chatbot.reset_conversation()
            continue
        
        if not user_input:
            continue
        
        response = chatbot.chat(user_input)
        print(f"\nBot: {response}\n")


if __name__ == "__main__":
    main()
