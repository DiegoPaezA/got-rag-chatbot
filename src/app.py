"""Streamlit web UI for the Game of Thrones RAG system."""

import streamlit as st
import os
import sys
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.rag.retriever import HybridRetriever
from src.rag.generator import RAGGenerator

# Page configuration
st.set_page_config(
    page_title="Maester AI - Game of Thrones",
    page_icon="🐉",
    layout="wide"
)

# Game of Thrones themed styling
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        color: #d4af37;
    }
    
    /* Overall text color */
    body {
        color: #c0c0c0;
        font-family: 'Georgia', serif;
    }
    
    /* Titles */
    h1, h2, h3 {
        color: #d4af37;
        font-family: 'Georgia', serif;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
        border-bottom: 2px solid #8b7355;
        padding-bottom: 10px;
    }
    
    /* Chat messages - User */
    .stChatMessage[data-testid="stChatMessageUser"] {
        background: linear-gradient(135deg, #1a3a52 0%, #2c5282 100%);
        border-left: 4px solid #d4af37;
        border-radius: 8px;
        padding: 15px 12px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
    }
    
    /* Chat messages - Assistant */
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background: linear-gradient(135deg, #2d1b1b 0%, #4a2c2c 100%);
        border-left: 4px solid #8b7355;
        border-radius: 8px;
        padding: 15px 12px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
    }
    
    /* Input box */
    .stChatInput {
        border: 2px solid #8b7355 !important;
        border-radius: 8px;
        background: #1a1f3a !important;
        color: #c0c0c0 !important;
        font-family: 'Georgia', serif;
    }
    
    .stChatInput input {
        color: #c0c0c0 !important;
        background: #1a1f3a !important;
    }
    
    /* Expander containers */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #2d1b1b 0%, #3a2a2a 100%) !important;
        border: 1px solid #8b7355 !important;
        border-radius: 6px !important;
        color: #d4af37 !important;
        font-weight: bold;
        padding: 10px;
    }
    
    .streamlit-expanderContent {
        background: #0a0e27 !important;
        border: 1px solid #8b7355 !important;
        border-top: none !important;
        border-radius: 0 0 6px 6px !important;
        padding: 15px;
    }
    
    /* Subheaders */
    .stSubheader {
        color: #d4af37 !important;
        font-size: 18px !important;
        font-weight: bold !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #8b7355 0%, #a0826d 100%) !important;
        color: #0a0e27 !important;
        border: 2px solid #d4af37 !important;
        border-radius: 6px !important;
        font-weight: bold !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #a0826d 0%, #8b7355 100%) !important;
        box-shadow: 0 0 15px rgba(212, 175, 55, 0.5) !important;
    }
    
    /* Spinner text */
    .stSpinner {
        color: #d4af37 !important;
    }
    
    /* Error boxes */
    .stAlert {
        background: linear-gradient(135deg, #3a1a1a 0%, #4a2a2a 100%) !important;
        border: 2px solid #d4af37 !important;
        border-radius: 6px !important;
        color: #ff6b6b !important;
    }
    
    /* JSON display */
    pre {
        background: #0f1420 !important;
        border: 1px solid #8b7355 !important;
        border-radius: 6px !important;
        color: #d4af37 !important;
        padding: 12px !important;
    }
    
    /* Divider lines */
    hr {
        border-color: #8b7355 !important;
        opacity: 0.5;
    }
</style>
""", unsafe_allow_html=True)

st.title("🐉 The Maester's Archives")
st.markdown("""
<div style='text-align: center; padding: 15px; border: 2px solid #d4af37; border-radius: 8px; 
            background: linear-gradient(135deg, #1a1f3a 0%, #2d1b1b 100%); margin-bottom: 20px;'>
    <p style='color: #d4af37; font-size: 16px; font-style: italic;'>
        "The Maester's knowledge spans the realm of Westeros and beyond..."
    </p>
    <p style='color: #c0c0c0;'>Ask about lineage, battles, and the history of the Seven Kingdoms.</p>
</div>
""", unsafe_allow_html=True)

# Lazy load engine components
@st.cache_resource
def load_engine():
    """Load retriever and generator with caching."""
    retriever = HybridRetriever()
    generator = RAGGenerator()
    return retriever, generator

try:
    retriever, generator = load_engine()
except Exception as e:
    st.error(f"❌ Error connecting to the archives: {e}")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("What is dead may never die..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Consulting the scrolls..."):
            try:
                context = retriever.retrieve(prompt)
                
# Show retrieved context in expandable sections
                with st.expander("🔍 What did the Maester find?", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🕸️ Graph Facts")
                        graph_data = context.get("graph_context")
                        if graph_data:
                            st.json(graph_data)
                        else:
                            st.markdown("""
                            <p style='color: #8b7355; font-style: italic;'>
                            No direct relationships found in the Great Ledger...
                            </p>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.subheader("📄 Scroll Fragments")
                        vector_docs = context.get("vector_context")
                        if vector_docs:
                            for i, doc in enumerate(vector_docs, 1):
                                st.markdown(f"""
                                <div style='background: #1a1f3a; padding: 10px; margin: 8px 0; 
                                           border-left: 3px solid #8b7355; border-radius: 4px;'>
                                    <p style='color: #d4af37; font-weight: bold;'>Fragment {i}</p>
                                    <p style='color: #c0c0c0; font-size: 13px;'>{doc[:300]}...</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <p style='color: #8b7355; font-style: italic;'>
                            No scrolls found in the Archives...
                            </p>
                            """, unsafe_allow_html=True)

                full_response = generator.generate_answer(prompt, context)
                message_placeholder.markdown(full_response)
                
            except Exception as e:
                full_response = f"The ravens are lost. Error: {e}"
                message_placeholder.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})