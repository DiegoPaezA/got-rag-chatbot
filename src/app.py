"""Streamlit web UI for the Game of Thrones RAG system."""

import streamlit as st
import os
import sys
import logging
import json

# Ensure project root is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.rag.retriever import HybridRetriever
from src.rag.augmenter import ContextAugmenter
from src.rag.generator import RAGGenerator

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Maester AI - Game of Thrones",
    page_icon="🐉",
    layout="wide"
)

# --- CSS STYLES ---
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        color: #d4af37;
    }
    
    /* Global text color */
    body {
        color: #c0c0c0;
        font-family: 'Georgia', serif;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #d4af37;
        font-family: 'Georgia', serif;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
        border-bottom: 2px solid #8b7355;
        padding-bottom: 10px;
    }
    
    /* User messages */
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #2c3e50;
        border-radius: 15px;
        border: 1px solid #8b7355;
    }
    
    /* Assistant messages */
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #1a1f3a;
        border-radius: 15px;
        border: 1px solid #d4af37;
    }

    /* Fragment cards */
    .fragment-card {
        background: #15192b; 
        padding: 10px; 
        margin: 8px 0; 
        border-left: 3px solid #8b7355; 
        border-radius: 4px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZATION (SINGLETON) ---
@st.cache_resource
def get_engine():
    """Initialize the 3 Pipeline components efficiently."""
    retriever = HybridRetriever()
    augmenter = ContextAugmenter()  # Loads its own LLM on startup
    generator = RAGGenerator()      # Pure text generator
    return retriever, augmenter, generator

try:
    retriever, augmenter, generator = get_engine()
except Exception as e:
    st.error(f"Failed to initialize RAG engine: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- MEMORY FUNCTION ---
def format_chat_history(messages: list, limit: int = 4) -> list:
    """Convert Streamlit chat history to the retriever-friendly format."""
    history = []
    recent = messages[-limit:]
    for msg in recent:
        role = "Human" if msg["role"] == "user" else "AI"
        history.append((role, msg["content"]))
    return history

# --- MAIN UI ---
st.title("🐉 Maester AI: The Citadel Archives")
st.markdown("*Ask questions about the history, lineage, and secrets of Westeros.*")

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input processing
if prompt := st.chat_input("What connects the Great Houses?"):
    
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Process assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            with st.spinner("Consulting the archives..."):
                
                # A) Prepare chat history
                history_tuples = format_chat_history(st.session_state.messages[:-1])
                
                # B) RETRIEVAL (Raw Data)
                # Retrieve raw data from the graph and vectors
                raw_context = retriever.retrieve(query=prompt, chat_history=history_tuples)
                
                # Important: Use the refined query (e.g., coreference resolved)
                refined_query = raw_context.get("refined_query", prompt)
                
                # --- Debug: Query Refinement ---
                if refined_query != prompt:
                    with st.expander("🧠 Maester's Thought Process (Query Refinement)"):
                        st.caption(f"Original: {prompt}")
                        st.markdown(f"**Searching for:** `{refined_query}`")       

                # --- Debug: Sources ---
                if raw_context.get("vector_context") or raw_context.get("graph_context"):
                    with st.expander("📜 Ancient Scrolls & Lineages (Raw Sources)", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        # Graph Column
                        with col1:
                            st.subheader("🕸️ Graph Knowledge")
                            st.caption("Structured data from Neo4j")
                            g_data = raw_context.get("graph_context", [])
                            if g_data:
                                # Show raw JSON for debugging
                                st.json(g_data, expanded=False)
                            else:
                                st.info("No direct graph connections found.")

                        # Vector Column
                        with col2:
                            st.subheader("📄 Text Fragments")
                            st.caption("Unstructured text from Vector DB")
                            v_docs = raw_context.get("vector_context", [])
                            if v_docs:
                                for i, doc in enumerate(v_docs[:3], 1):
                                    st.markdown(f"**Fragment {i}:** {doc[:200]}...")
                            else:
                                st.info("No text fragments found.")

                # C) AUGMENTATION (The Analyst)
                # Transform raw data into a narrative intelligence report
                formatted_context_str = augmenter.build_context(
                    query=refined_query,  # CRITICAL: Pass the refined query so the LLM focuses the report
                    vector_context=raw_context.get("vector_context", []),
                    graph_context=raw_context.get("graph_context", [])
                )
                
                # Show the generated report to understand what the final generator 'reads'
                with st.expander("🧐 Intelligence Report (Formatted Context)", expanded=False):
                    st.markdown(formatted_context_str)
                
                # D) GENERATION (The Maester)
                # The final generator only receives the formatted report and the question
                full_response = generator.generate_answer(
                    question=refined_query, 
                    formatted_context=formatted_context_str
                )
                
                message_placeholder.markdown(full_response)
            
        except Exception as e:
            st.error(f"The ravens are lost. Error: {e}")
            full_response = "I am unable to answer at this moment."

    st.session_state.messages.append({"role": "assistant", "content": full_response})