import pandas as pd
import re
from typing import List, TypedDict, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer, util
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Initialize the unsupervised model globally ---
try:
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    semantic_model = None

# --- 1. Define the State for our Graph ---
class RefineryState(TypedDict):
    report_data: Dict[str, Any]
    cleaned_text: str
    macro_chunks: List[str]
    micro_chunks: List[Dict[str, Any]]
    final_output: List[Dict[str, Any]]

# --- 2. Define the Nodes of our Graph ---
def ingest_and_preprocess(state: RefineryState) -> Dict[str, Any]:
    report = state['report_data']
    symptom_text = report.get('SYMPTOM_TEXT', '')
    text = symptom_text.lower()
    boilerplate = [
        r"patient states:", r"information was received from...", r"symptoms:",
        r"write-up:", r"patient reports:", r"no adverse event was reported"
    ]
    for pattern in boilerplate:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return {"cleaned_text": text}

def semantic_chunking_node(state: RefineryState) -> Dict[str, Any]:
    text = state['cleaned_text']
    if not semantic_model or not text:
        sentences = [s.strip() for s in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text) if s.strip()]
        return {"macro_chunks": sentences}

    sentences = [s.strip() for s in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text) if s.strip()]
    
    if len(sentences) < 2:
        return {"macro_chunks": sentences}

    embeddings = semantic_model.encode(sentences, convert_to_tensor=True)
    similarities = [util.pytorch_cos_sim(embeddings[i], embeddings[i+1]).item() for i in range(len(sentences) - 1)]
    split_points = [i + 1 for i, sim in enumerate(similarities) if sim < 0.3]

    semantic_chunks = []
    start_idx = 0
    for point in split_points:
        semantic_chunks.append(" ".join(sentences[start_idx:point]))
        start_idx = point
    semantic_chunks.append(" ".join(sentences[start_idx:]))
    
    return {"macro_chunks": semantic_chunks}

def enrich_chunks(state: RefineryState) -> Dict[str, Any]:
    report = state['report_data']
    chunks = state['macro_chunks']
    enriched_chunks = []
    for i, chunk_text in enumerate(chunks):
        weak_labels = [report.get(f'SYMPTOM{j}') for j in range(1, 6)]
        enriched_chunk = {
            "VAERS_ID": report.get('VAERS_ID'),
            "chunk_id": f"{report.get('VAERS_ID')}_{i}",
            "text": chunk_text,
            "age_yrs": report.get('AGE_YRS'),
            "sex": report.get('SEX'),
            "vax_manu": report.get('VAX_MANU'),
            "weak_labels": [label for label in weak_labels if pd.notna(label)]
        }
        enriched_chunks.append(enriched_chunk)
    return {"micro_chunks": enriched_chunks}

def quality_gate(state: RefineryState) -> Dict[str, Any]:
    chunks = state['micro_chunks']
    approved_chunks = [chunk for chunk in chunks if len(chunk['text'].split()) > 3]
    return {"final_output": approved_chunks}

# --- 3. Assemble the Graph ---
workflow = StateGraph(RefineryState)
workflow.add_node("ingest_and_preprocess", ingest_and_preprocess)
workflow.add_node("semantic_chunking", semantic_chunking_node)
workflow.add_node("enrich_chunks", enrich_chunks)
workflow.add_node("quality_gate", quality_gate)
workflow.set_entry_point("ingest_and_preprocess")
workflow.add_edge("ingest_and_preprocess", "semantic_chunking")
workflow.add_edge("semantic_chunking", "enrich_chunks")
workflow.add_edge("enrich_chunks", "quality_gate")
workflow.add_edge("quality_gate", END) 
app = workflow.compile()

# --- 4. Main Execution Logic ---
if __name__ == "__main__":
    INPUT_FILE = 'covid_vaers_cleaned.csv'
    OUTPUT_FILE = 'refined_data.jsonl'
    # --- UPDATE: Increased sample size to generate more data ---
    SAMPLE_SIZE = 400
    
    try:
        df = pd.read_csv(INPUT_FILE)
        # --- UPDATE: Use the SAMPLE_SIZE variable ---
        sample_df = df.drop_duplicates(subset=['VAERS_ID']).sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        exit()

    print(f"Starting LangGraph Data Refinery on a sample of {len(sample_df)} reports...")
    final_refined_data = []
    for index, row in sample_df.iterrows():
        report_dict = row.to_dict()
        graph_input = {"report_data": report_dict}
        result = app.invoke(graph_input)
        if result.get('final_output'):
            final_refined_data.extend(result['final_output'])

    if final_refined_data:
        pd.DataFrame(final_refined_data).to_json(OUTPUT_FILE, orient='records', lines=True)
        print(f"\nRefinery process complete. {len(final_refined_data)} refined chunks saved to '{OUTPUT_FILE}'")
    else:
        print("\nRefinery process complete. No data was generated.")



