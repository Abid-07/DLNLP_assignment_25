import re
import random
import nltk
import spacy
from datasets import load_dataset, Dataset
from A.utils import (
    clean_long_quotes, tag_entities_with_spacy,
    map_legal_to_structured_text, compute_ecp,
    get_ecp_tag, get_section_tag,
    extract_summary_terms, score_sentences_by_summary_terms,
    tag_sentences_by_ecp
)
from transformers import AutoTokenizer

# Download required NLTK tokenizers and load spaCy model
nltk.download("punkt")
nltk.download("punkt_tab")
nlp = spacy.load("en_core_web_sm")

def generate_updated_mapping(example):
    """
    Applies structural and entity tagging to legal text and 
    aligns those changes with the summary.
    """
    mapped_text, entity_map = map_legal_to_structured_text(example["text"], return_entity_map=True)
    summary_text = example["summary"]

    for original, placeholder in entity_map.items():
        pattern = re.compile(rf"\b{re.escape(original)}\b", flags=re.IGNORECASE)
        summary_text = pattern.sub(placeholder, summary_text)

    return {
        "plain_view": example["text"],
        "mapped_view": mapped_text,
        "summary": summary_text
    }

def expand_dataset(dataset):
    """
    Adds control tokens and tags for section size and entity coverage precision (ECP),
    preparing examples for controlled summarization.
    """
    rows = []
    for ex in dataset:
        input_text = ex.get("mapped_view", ex.get("plain_view", ex.get("text", "")))
        summary_text = ex["summary"]

        ecp = compute_ecp(input_text, summary_text)
        ecp_tag = get_ecp_tag(ecp)
        sec_tag = get_section_tag(input_text)

        highlighted_input = tag_sentences_by_ecp(input_text, summary_text)
        highlighted_input = re.sub(r'\[PATH=.*?\]', '', highlighted_input)

        control_prefix = f"summarize: {ecp_tag} {sec_tag}"
        full_input = f"{control_prefix}\n{highlighted_input}"

        rows.append({
            "input": full_input,
            "label": summary_text,
            "ecp_score": ecp
        })

    return Dataset.from_list(rows)

def preprocess_batch(batch, tokenizer, max_input_length=1024, max_target_length=512):
    """
    Tokenizes inputs and labels for the model, and computes 
    sample weights based on inverse ECP score.
    """
    inputs = batch["input"]
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
        return_special_tokens_mask=True,
    )
    labels = tokenizer(
        text_target=batch["label"],
        max_length=max_target_length,
        padding="max_length",
        truncation=True,
    )["input_ids"]

    weights = [1.0 + (1.0 - score) for score in batch["ecp_score"]]

    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": labels,
        "sample_weights": weights
    }

def get_datasets():
    """
    Loads the BillSum dataset, applies mapping and tagging, 
    tokenizes the data, and returns training, validation, and test sets.
    """
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
    dataset = load_dataset("billsum")

    train_dataset = dataset['train'].select(range(4000)).map(generate_updated_mapping)
    expanded_dataset = expand_dataset(train_dataset)
    expanded_dataset = expanded_dataset.add_column("original_idx", list(range(len(expanded_dataset))))

    tokenized_dataset = expanded_dataset.map(
        lambda batch: preprocess_batch(batch, tokenizer),
        batched=True,
        remove_columns=["input", "label"]
    )

    split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_base = split["train"]
    val_data = split["test"]

    test_raw = dataset['test'].select(range(500)).map(generate_updated_mapping)
    expanded_test_dataset = expand_dataset(test_raw)
    expanded_test_dataset = expanded_test_dataset.add_column("original_idx", list(range(len(expanded_test_dataset))))

    return train_base, val_data, expanded_test_dataset
