import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
MODEL_PATH = "./ade-ner-model"
SAMPLE_TEXT = "The patient reported a severe headache and nausea approximately 2 hours after receiving the Pfizer vaccine."

def main():
    """
    Main function to load the fine-tuned NER model and run inference to extract
    full entities from a sample text.
    """
    # 1. Load the fine-tuned model and tokenizer
    try:
        logging.info(f"Loading model from {MODEL_PATH}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
        logging.info("Model and tokenizer loaded successfully.")
    except OSError:
        logging.error(f"Model not found at '{MODEL_PATH}'.")
        logging.error("Please ensure you have run the 'train_ner_model.py' script successfully.")
        return

    # 2. Create a Hugging Face NER pipeline for clean, grouped output
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        # --- FINAL FIX: Using a more robust aggregation strategy ---
        aggregation_strategy="first"
    )

    # 3. Run inference on the sample text
    logging.info(f"Running inference on text: '{SAMPLE_TEXT}'")
    entities = ner_pipeline(SAMPLE_TEXT)

    # 4. Display the final, user-friendly results
    print("\n--- NER Inference Results ---")
    if entities:
        print(f"Found {len(entities)} entities:")
        for entity in entities:
            entity_word = entity['word'].strip()
            entity_type = entity['entity_group']
            entity_score = entity['score']
            
            print(f"  - Entity: '{entity_word}'")
            print(f"    Type: {entity_type}")
            print(f"    Confidence: {entity_score:.4f}")
    else:
        print("No entities were found in the text.")
    print("---------------------------\n")


if __name__ == "__main__":
    main()
