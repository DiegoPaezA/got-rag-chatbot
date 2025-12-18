import streamlit as st
import json
import os
import pandas as pd
import random
from datetime import datetime

# --- CONFIGURATION ---
DATA_DIR = "data/processed"
NODES_PATH = os.path.join(DATA_DIR, "nodes_validated.jsonl")  # Or nodes.jsonl if the LLM was not used yet
GOLD_DIR = "data/gold"
GOLD_FILE = os.path.join(GOLD_DIR, "gold_labels.jsonl")

# Allowed types (same as config)
ALLOWED_TYPES = [
    "Character", "House", "Organization", "Battle",
    "Location", "Object", "Creature", "Religion", "Episode", "Lore"
]

# Page settings
st.set_page_config(page_title="GoT Graph Validator", layout="wide")

# --- FUNCIONES ---

def load_data():
    """Load nodes and prepare or recover the stratified sample."""
    if not os.path.exists(NODES_PATH):
        st.error(f"❌ No se encontró {NODES_PATH}")
        return []

    # Cargamos todos los nodos
    data = []
    with open(NODES_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    return df

def get_stratified_sample(df, n=200):
    """Select n examples while balancing classes instead of pure random sampling."""
    # Reuse an existing session sample to keep nodes stable between reloads
    sample_file = os.path.join(GOLD_DIR, "validation_queue.csv")
    
    if os.path.exists(sample_file):
        return pd.read_csv(sample_file)
    
    st.toast("Generando nueva muestra estratificada...")
    
    # Strategy: take even samples per type
    types = df['type'].unique()
    samples_per_type = n // len(types)
    
    sampled_dfs = []
    for t in types:
        # Take X from each type, or all if the group is small
        subset = df[df['type'] == t]
        n_subset = min(len(subset), samples_per_type)
        if n_subset > 0:
            sampled_dfs.append(subset.sample(n=n_subset, random_state=42))
    
    # Concatenate
    sample = pd.concat(sampled_dfs)
    
    # If still short of n, fill randomly from the remainder
    remaining = n - len(sample)
    if remaining > 0:
        rest = df[~df['id'].isin(sample['id'])]
        if len(rest) > 0:
            sample = pd.concat([sample, rest.sample(n=min(len(rest), remaining), random_state=42)])
    
    # Shuffle and persist
    sample = sample.sample(frac=1).reset_index(drop=True)
    
    if not os.path.exists(GOLD_DIR):
        os.makedirs(GOLD_DIR)
    
    # Save only IDs and key data for the queue
    sample[['id', 'type', 'url', 'properties']].to_csv(sample_file, index=False)
    
    return sample

def load_existing_gold():
    """Load validated items to avoid repetition."""
    if not os.path.exists(GOLD_FILE):
        return {}
    
    gold_map = {}
    with open(GOLD_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                gold_map[item['id']] = item['gold_type']
            except: continue
    return gold_map

def save_gold_label(node_id, gold_type, comments=""):
    """Persist a gold label to the JSONL file (append mode)."""
    record = {
        "id": node_id, 
        "gold_type": gold_type, 
        "timestamp": datetime.now().isoformat(),
        "comments": comments
    }
    
    # Append to the file
    with open(GOLD_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# --- MAIN INTERFACE ---

st.title("🛡️ The Watchers on the Wall: Graph Validator")
st.markdown("Manual validation tool to create the **Gold Standard**.")

# 1. Cargar Datos
df_all = load_data()
if df_all.empty:
    st.stop()

df_sample = get_stratified_sample(df_all, n=200)
gold_map = load_existing_gold()

# 2. Filter pending entries (not present in gold_map)
pending_df = df_sample[~df_sample['id'].isin(gold_map.keys())].reset_index(drop=True)
completed_count = len(gold_map)
total_target = len(df_sample)

# Progress bar
progress = completed_count / total_target
st.progress(progress, text=f"Progreso: {completed_count}/{total_target} entidades validadas")

if pending_df.empty:
    st.success("🎉 The watch has ended! You validated all entities in the sample.")
    st.balloons()
    
    # Mostrar botón para descargar
    with open(GOLD_FILE, "r") as f:
        st.download_button("Descargar Gold Labels", f, file_name="gold_labels.jsonl")
    st.stop()

# 3. Show Current Entity (the first pending one)
current_node = pending_df.iloc[0]

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"⚔️ {current_node['id']}")
    
    # Link to the Wiki
    if current_node.get('url'):
        st.markdown(f"🔗 [Abrir en Wiki Fandom]({current_node['url']}) (Revisa la fuente real)")
    
    # Current info
    st.info(f"**Detected Type:** {current_node['type']}")
    
    # Properties (Infobox)
    st.markdown("#### 📜 Extracted properties:")
    props = eval(str(current_node['properties'])) if isinstance(current_node['properties'], str) else current_node['properties']
    st.json(props)

with col2:
    st.markdown("### 🏷️ Human Validation")
    
    # Formulario
    with st.form("validation_form"):
        # Correct type selector; default to detected type if present
        default_idx = ALLOWED_TYPES.index(current_node['type']) if current_node['type'] in ALLOWED_TYPES else ALLOWED_TYPES.index("Lore")
        
        selected_type = st.selectbox(
            "What is the CORRECT category?", 
            options=ALLOWED_TYPES,
            index=default_idx
        )
        
        comments = st.text_input("Comments (optional)", placeholder="e.g., Ambiguous, but counts as Battle")
        
        submitted = st.form_submit_button("✅ Confirm and Next", type="primary")
        
        if submitted:
            save_gold_label(current_node['id'], selected_type, comments)
            st.toast(f"Saved: {current_node['id']} -> {selected_type}")
            st.rerun() # Recargar para pasar al siguiente

# Mostrar historial reciente en sidebar
with st.sidebar:
    st.header("Recent History")
    if os.path.exists(GOLD_FILE):
        hist_df = pd.read_json(GOLD_FILE, lines=True).tail(10).iloc[::-1]
        st.dataframe(hist_df[['id', 'gold_type']], hide_index=True)