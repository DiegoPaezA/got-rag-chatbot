"""Streamlit web UI for the Game of Thrones RAG system."""

import streamlit as st
import os
import sys
import logging
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.rag.retriever import HybridRetriever
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
    return HybridRetriever(), RAGGenerator()

try:
    retriever, generator = get_engine()
except Exception as e:
    st.error(f"Failed to initialize RAG engine: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- MEMORY FUNCTION ---
def format_chat_history(messages: list, limit: int = 4) -> list:
    """Convert Streamlit chat history to the retriever-friendly format."""
    history = []
    # Skip the latest message because it is the current prompt we just added
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

# Input
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
                
                # A) Prepare chat history for memory (excluding the current prompt)
                history_tuples = format_chat_history(st.session_state.messages[:-1])
                
                # B) Retrieve with memory
                context = retriever.retrieve(query=prompt, chat_history=history_tuples)
                print(json.dumps(context, indent=2))  # Console debug
                # C) Debug: show if the question was rewritten
                if "refined_query" in context and context["refined_query"] != prompt:
                    with st.expander("🧠 Maester's Thought Process (Query Refinement)"):
                        st.caption(f"Original: {prompt}")
                        st.markdown(f"**Searching for:** `{context['refined_query']}`")

                # D) Source visualization: graph vs text
                if context.get("vector_context") or context.get("graph_context"):
                    with st.expander("📜 Ancient Scrolls & Lineages (Sources)", expanded=False):
                        col1, col2 = st.columns(2)
                        # --- COLUMN 1: GRAPH KNOWLEDGE ---
                        with col1:
                            st.subheader("🕸️ Graph Knowledge")
                            st.caption("Structured data from Neo4j")
                            
                            g_data = context.get("graph_context", [])
                            
                            if g_data:
                                for node in g_data:
                                    with st.container():
                                        # Visual header to separate each matched node
                                        st.markdown("#### 🔹 Entity Match")
                                        
                                        # Iterate over every property returned for the node
                                        for key, value in node.items():
                                            # Cleanup: "c.name" -> "name"
                                            clean_key = key.split(".")[-1] if "." in key else key
                                            
                                            # Render as list item
                                            st.markdown(f"- **{clean_key}:** {value}")
                                        
                                        st.divider()
                            else:
                                st.info("No direct graph connections found.")

                        # --- COLUMN 2: TEXT FRAGMENTS ---
                        with col2:
                            st.subheader("📄 Text Fragments")
                            st.caption("Unstructured text from Vector DB")
                            
                            v_docs = context.get("vector_context", [])
                            
                            if v_docs:
                                for i, doc in enumerate(v_docs[:4], 1):
                                    st.markdown(f"""
                                    <div class="fragment-card">
                                        <div style='color: #d4af37; font-weight: bold; font-size: 0.8em; margin-bottom:5px;'>
                                            📜 Fragment {i}
                                        </div>
                                        <div style='color: #e0e0e0; font-size: 0.85em; line-height: 1.4;'>
                                            {doc[:400]}...
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.info("No text fragments found.")

                # E) Generation
                full_response = generator.generate_answer(prompt, context)
                message_placeholder.markdown(full_response)
            
        except Exception as e:
            st.error(f"The ravens are lost. Error: {e}")
            full_response = "I am unable to answer at this moment."

    st.session_state.messages.append({"role": "assistant", "content": full_response})