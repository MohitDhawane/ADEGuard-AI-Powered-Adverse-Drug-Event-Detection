from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
MODEL_PATH = "./ade-ner-model"

# --- Pydantic Models for Request and Response ---
class NERRequest(BaseModel):
    text: str

class Entity(BaseModel):
    word: str
    type: str
    confidence: float

class NERResponse(BaseModel):
    entities: list[Entity]

# --- FastAPI Application ---
app = FastAPI(
    title="ADEGuard NER API",
    description="An API to extract Adverse Drug Events (ADEs) and DRUGs from text.",
    version="1.0.0"
)

# --- Global Model Loading ---
# Load the model once when the application starts to avoid reloading on every request.
ner_pipeline = None

@app.on_event("startup")
def load_model():
    """Load the NER pipeline at application startup."""
    global ner_pipeline
    try:
        logging.info(f"Loading NER model from {MODEL_PATH}...")
        ner_pipeline = pipeline(
            "ner",
            model=MODEL_PATH,
            tokenizer=MODEL_PATH,
            aggregation_strategy="first" # Use the robust aggregation strategy
        )
        logging.info("NER model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load NER model: {e}")
        ner_pipeline = None # Ensure pipeline is None if loading fails

# --- API Endpoint ---
@app.post("/extract-entities", response_model=NERResponse)
def extract_entities(request: NERRequest):
    """
    Accepts a text string and returns a list of extracted entities (ADEs and DRUGs).
    """
    if ner_pipeline is None:
        raise HTTPException(status_code=503, detail="NER model is not available or failed to load.")

    try:
        logging.info(f"Received request for text: '{request.text}'")
        raw_entities = ner_pipeline(request.text)
        
        # Format the entities to match our response model
        formatted_entities = [
            Entity(
                word=entity['word'].strip(),
                type=entity['entity_group'],
                confidence=round(entity['score'], 4)
            ) for entity in raw_entities
        ]
        
        return NERResponse(entities=formatted_entities)

    except Exception as e:
        logging.error(f"An error occurred during inference: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during entity extraction.")

# To run this API:
# 1. Install FastAPI and Uvicorn: pip install "fastapi[all]"
# 2. Run from your terminal:  uvicorn api_backend:app --reload
