# train_ner_model.py
# Fine-tune a token classification (NER) model on tokens + integer tags.
# Compatible with older Transformers that don't accept evaluation/save strategy args.

import os
import sys
import json
import logging
from typing import List, Dict

from datasets import Dataset, Features, Sequence, ClassLabel, Value
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

# -------------------------
# Config (adjust as needed)
# -------------------------
MODEL_CHECKPOINT   = "dmis-lab/biobert-v1.1"
INPUT_DATA_FILE    = "ner_training_data_hf.json"   # expects: [{"tokens":[...], "ner_tags":[ints]}, ...]
LABELS_FILE        = "labels.json"                 # optional; falls back if missing
MODEL_OUTPUT_DIR   = "./ade-ner-model"

TRAIN_TEST_SPLIT   = 0.20
LEARNING_RATE      = 2e-5
EPOCHS             = 10
BATCH_SIZE         = 4
WEIGHT_DECAY       = 0.01
SEED               = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_data(path: str = INPUT_DATA_FILE) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logging.info(f"Loaded {len(data)} examples from {path}")
    if not data or "tokens" not in data[0] or "ner_tags" not in data[0]:
        sys.exit("Dataset must contain 'tokens' (list[str]) and 'ner_tags' (list[int]) per example.")
    return data


def load_labels() -> tuple[list[str], dict[int, str], dict[str, int]]:
    default_labels = ["B-ADE", "B-DRUG", "I-ADE", "I-DRUG", "O"]
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            labels = json.load(f)
        logging.info(f"Label order loaded from {LABELS_FILE}: {labels}")
    else:
        labels = default_labels
        logging.info(f"No {LABELS_FILE} found; using fallback label order: {labels}")
    id2label = {i: lab for i, lab in enumerate(labels)}
    label2id = {lab: i for i, lab in enumerate(labels)}
    return labels, id2label, label2id


def build_dataset(rows: List[Dict], labels: List[str]) -> Dataset:
    # Keep columns precise: only tokens (strings) and ner_tags (ints)
    feats = Features({
        "tokens":   Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=labels)),
    })
    ds = Dataset.from_dict(
        {
            "tokens":   [r["tokens"]   for r in rows],
            "ner_tags": [r["ner_tags"] for r in rows],
        },
        features=feats,
    )
    return ds


def tokenize_and_align_labels(examples, tokenizer):
    # Tokenize as split-into-words and align integer tags to subwords
    tokenized = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True)
    aligned = []
    for i in range(len(examples["tokens"])):
        word_ids = tokenized.word_ids(batch_index=i)
        word_labels = examples["ner_tags"][i]
        labels = []
        prev = None
        for wid in word_ids:
            if wid is None:
                labels.append(-100)  # special tokens
            elif wid != prev:
                labels.append(word_labels[wid])  # first subword gets the label
            else:
                labels.append(-100)  # mask subsequent subwords
            prev = wid
        aligned.append(labels)
    tokenized["labels"] = aligned
    return tokenized


def main():
    rows = load_data()
    labels, id2label, label2id = load_labels()

    # Sanity: verify tag ids are within label space
    max_tag = max((max(r["ner_tags"]) for r in rows if r["ner_tags"]), default=0)
    if max_tag >= len(labels):
        sys.exit(
            f"Found tag id {max_tag} but only {len(labels)} labels provided. "
            f"Ensure your ner_tags ids match labels.json order."
        )

    # Build HF dataset and split
    ds = build_dataset(rows, labels)
    split = ds.train_test_split(test_size=TRAIN_TEST_SPLIT, seed=SEED)
    logging.info(f"Train={len(split['train'])}, Test={len(split['test'])}")

    # Tokenizer & token-level alignment
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    tokenized = split.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer),
        batched=True,
        remove_columns=ds.column_names,
        desc="Tokenize+align",
    )

    # Model
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    # NOTE: Keep TrainingArguments minimal for compatibility with older Transformers.
    args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        logging_dir="./logs",
        seed=SEED,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        # (Optional) add eval_dataset and metrics once your environment supports them
    )

    logging.info("Starting training…")
    trainer.train()
    logging.info("Training complete.")

    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    logging.info(f"Saved model & tokenizer to {MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()






