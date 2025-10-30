"""
Streamlit UI for Tech Support Chatbot
Provides an interactive web interface for the chatbot
"""

import streamlit as st
from chatbot import TechSupportChatbot
import time


# Page configuration
st.set_page_config(
    page_title="Tech Support Chatbot",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        color: #000000;
    }
    .bot-message {
        background-color: #f5f5f5;
        border-left: 5px solid #4caf50;
        color: #000000;
    }
    .message-label {
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_chatbot():
    """Load and cache the chatbot model."""
    return TechSupportChatbot()


def display_message(role: str, message: str):
    """Display a chat message with styling."""
    if role == "user":
        st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-label">You</div>
                <div style="color: #000000;">{message}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="chat-message bot-message">
                <div class="message-label">Tech Support Bot</div>
                <div style="color: #000000;">{message}</div>
            </div>
        """, unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("Tech Support Chatbot")
    st.markdown("*Your AI-powered technical support assistant*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This chatbot provides technical support for **TechSoft Pro Suite**.
        
        **Features:**
        - Local LLM (runs entirely offline)
        - Multi-turn conversation support
        - Context-aware responses
        - Semantic search over knowledge base
        
        **Coverage:**
        - Account management
        - Billing & subscriptions
        - Technical troubleshooting
        - Feature questions
        - Security & privacy
        """)
        
        st.markdown("---")
        
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            if 'chatbot' in st.session_state:
                st.session_state.chatbot.reset_conversation()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Stats")
        if 'messages' in st.session_state:
            st.metric("Messages", len(st.session_state.messages))
        else:
            st.metric("Messages", 0)
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Loading chatbot... This may take a moment on first run."):
            st.session_state.chatbot = load_chatbot()
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        welcome_msg = (
            "Hello! I'm your Tech Support assistant. "
            "I can help you with questions about TechSoft Pro Suite, including:\n\n"
            "• Account management\n"
            "• Billing and subscriptions\n"
            "• Technical issues\n"
            "• Features and capabilities\n"
            "• Security questions\n\n"
            "How can I assist you today?"
        )
        st.session_state.messages.append({
            "role": "assistant",
            "content": welcome_msg
        })
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            display_message(message["role"], message["content"])
    
    # Chat input
    st.markdown("---")
    user_input = st.chat_input("Type your question here...", key="user_input")
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Display user message
        with chat_container:
            display_message("user", user_input)
        
        # Generate bot response
        with st.spinner("Thinking..."):
            response = st.session_state.chatbot.chat(user_input)
        
        # Add bot response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        
        # Display bot response
        with chat_container:
            display_message("assistant", response)
        
        # Rerun to update the display
        st.rerun()
    
    # Example questions
    st.markdown("---")
    st.markdown("### Try asking:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("How do I reset my password?", use_container_width=True):
            st.session_state.next_input = "How do I reset my password?"
            st.rerun()
    
    with col2:
        if st.button("What's included in Pro plan?", use_container_width=True):
            st.session_state.next_input = "What's included in Pro plan?"
            st.rerun()
    
    with col3:
        if st.button("How secure is my data?", use_container_width=True):
            st.session_state.next_input = "How secure is my data?"
            st.rerun()
    
    # Handle example question clicks
    if 'next_input' in st.session_state:
        user_question = st.session_state.next_input
        del st.session_state.next_input
        
        # Add to messages and get response
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })
        
        response = st.session_state.chatbot.chat(user_question)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        
        st.rerun()


if __name__ == "__main__":
    main()
