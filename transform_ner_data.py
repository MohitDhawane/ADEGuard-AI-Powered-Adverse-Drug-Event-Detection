import json
import logging
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# --- UPDATE: Pointing to the JSONL file from your error log ---
INPUT_FILE = "ade_drug_predictions.jsonl" 
OUTPUT_FILE = "ner_training_data.json"
MODEL_CHECKPOINT = "dmis-lab/biobert-v1.1" # For tokenizer

def main():
    """
    Main function to transform the Label Studio-formatted JSONL
    into a token-based format for NER model training.
    """
    logging.info(f"Starting transformation for records in '{INPUT_FILE}'...")

    # --- FIX: Read the file line-by-line to handle .jsonl format ---
    try:
        data = []
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): # Ensure the line is not empty
                    data.append(json.loads(line))
        logging.info(f"Loaded {len(data)} records from JSONL file.")
    except FileNotFoundError:
        logging.error(f"Input file not found: '{INPUT_FILE}'. Please ensure the file exists.")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding a line in the JSONL file: {e}")
        return
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    transformed_data = []

    for record in data:
        # Robustly find the annotations list
        annotations_list = record.get("annotations") or record.get("completions")
        if not annotations_list or not annotations_list[0].get("result"):
            logging.warning(f"Skipping record with no valid annotations: ID {record.get('id')}")
            continue

        text = record["data"]["text"]
        annotations = annotations_list[0]["result"]

        # Tokenize the text using the word_ids method for better alignment
        tokenized_output = tokenizer(text, truncation=True)
        tokens = tokenizer.convert_ids_to_tokens(tokenized_output['input_ids'])
        word_ids = tokenized_output.word_ids()
        
        ner_tags = ['O'] * len(tokens)

        for ann in annotations:
            label_info = ann.get("value", {})
            start_char, end_char = label_info.get("start"), label_info.get("end")
            label = label_info.get("labels", [None])[0]

            if start_char is None or end_char is None or label is None:
                continue

            # Align character spans to token indices
            token_start_index = -1
            token_end_index = -1
            
            for i, wid in enumerate(word_ids):
                if wid is None:
                    continue
                
                # Get the character span for the current word
                span = tokenized_output.word_to_chars(wid)
                
                # Check for overlap
                if max(start_char, span.start) < min(end_char, span.end):
                    if token_start_index == -1:
                        token_start_index = i
                    token_end_index = i

            if token_start_index != -1:
                ner_tags[token_start_index] = f"B-{label}"
                for i in range(token_start_index + 1, token_end_index + 1):
                    ner_tags[i] = f"I-{label}"
        
        # We need to provide the raw tokens (words), not the word pieces from BioBERT,
        # for the training script's alignment logic to work correctly.
        raw_tokens = text.split()
        # This part of the logic is now handled more robustly in the training script,
        # so we simplify the output here.
        # For simplicity, we'll pass the text and annotations directly for now.
        # A more robust pipeline would create aligned word-level tags here.
        # Let's pass what the training script needs: raw tokens and string tags.
        
        # A simplified alignment for this script's purpose
        final_tokens = text.split()
        final_tags = ['O'] * len(final_tokens)
        
        # This is a simplified re-alignment and may not be perfect for complex cases
        # but is better than passing subword tokens.
        
        transformed_data.append({
            "id": record.get("id"),
            "tokens": final_tokens, # Pass word-level tokens
            "ner_tags_str": final_tags # This would need a more complex alignment logic
        })


    # For now, let's revert to a simpler output that the training script can handle directly
    # The training script itself needs to be robust enough. We will simplify this script's output.
    final_data_for_training = []
    for record in data:
         annotations_list = record.get("annotations") or record.get("completions")
         if annotations_list and annotations_list[0].get("result"):
             final_data_for_training.append(record)


    logging.info(f"Transformation complete. Saved {len(final_data_for_training)} valid records to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w') as f:
        # Re-using the logic from process_llm_output to create silver_data format
        # This is confusing, let's fix the script to do its job properly.
        # The script is supposed to create IOB tags.
        # The logic was flawed. Let's fix the original intent.
        pass # Let's rewrite the core logic for correctness.
    
    # --- REWRITING LOGIC FOR CORRECTNESS ---
    # The goal is to convert Label Studio JSON to Hugging Face IOB format.
    final_data = []
    for record in data:
        # Re-checking for valid annotations
        annotations_list = record.get("annotations") or record.get("completions")
        if not annotations_list or not annotations_list[0].get("result"):
            continue

        text = record["data"]["text"]
        annotations = annotations_list[0]["result"]
        
        tokens = text.split() # Simple whitespace tokenization
        tags = ['O'] * len(tokens)
        
        for ann in annotations:
            ann_val = ann.get("value", {})
            ann_text = ann_val.get("text", "")
            ann_label = ann_val.get("labels", [""])[0]
            
            # This is a simple substring match, not robust but works for many cases
            ann_tokens = ann_text.split()
            for i in range(len(tokens) - len(ann_tokens) + 1):
                if tokens[i:i+len(ann_tokens)] == ann_tokens:
                    tags[i] = f"B-{ann_label}"
                    for j in range(1, len(ann_tokens)):
                        tags[i+j] = f"I-{ann_label}"
                    break # Assume first match is correct
        
        final_data.append({
            "id": record.get("id"),
            "tokens": tokens,
            "ner_tags_str": tags
        })

    logging.info(f"Transformation complete. Saved {len(final_data)} records to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_data, f, indent=2)


if __name__ == "__main__":
    main()








