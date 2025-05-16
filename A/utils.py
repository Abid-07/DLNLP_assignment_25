# Text preprocessing and training utilities for legal text summarization

import re
import random
import spacy
import nltk
import torch
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer
from summac.model_summac import SummaCConv
from transformers import TrainerCallback

nlp = spacy.load("en_core_web_sm")

# ======== TEXT PREPROCESSING UTILITIES ========

def clean_long_quotes(text, max_words_inside_quote=12):
    """
    Removes quotes that are too long from the text.
    """
    pattern = r'(.*?)'
    def replace(match):
        inside_text = match.group(1)
        if len(inside_text.split()) <= max_words_inside_quote:
            return f'{inside_text}'
        else:
            return ''
    return re.sub(pattern, replace, text)

def tag_entities_with_spacy(text, max_tags_per_sentence=1, mask_probability=0.20, return_map=False):
    """
    Identifies named entities using spaCy and replaces them with canonical placeholders.
    Returns a map of original entity to placeholder if return_map=True.
    """
    doc = nlp(text)
    spans = sorted(doc.ents, key=lambda e: -len(e.text))
    already_tagged = set()
    excluded_types = {"CARDINAL", "ORDINAL", "PERCENT", "QUANTITY", "MONEY", "TIME", "NORP"}

    entity_counters = {}
    entity_map = {}
    sentence_map = {}

    for sent in doc.sents:
        sent_text = sent.text
        tags_added = 0
        for ent in spans:
            if ent.label_ in excluded_types or ent.start < sent.start or ent.end > sent.end:
                continue

            ent_text = ent.text.strip()
            ent_label = ent.label_

            if not ent_text or any(c in ent_text for c in "[]") or \
               ent_text.lower() in nlp.Defaults.stop_words or \
               random.random() > mask_probability:
                continue

            if tags_added >= max_tags_per_sentence or ent_text.lower() in already_tagged:
                continue

            entity_counters.setdefault(ent_label, 0)
            entity_counters[ent_label] += 1
            placeholder = f"@{ent_label}_{entity_counters[ent_label]}@"
            already_tagged.add(ent_text.lower())
            entity_map[ent_text] = placeholder

            try:
                pattern = re.compile(rf"(?<!@)\b{re.escape(ent_text)}\b(?!@)", flags=re.IGNORECASE)
                sent_text = pattern.sub(placeholder, sent_text)
                tags_added += 1
            except re.error:
                continue

        sentence_map[sent.start_char] = sent_text

    updated_text = [sentence_map.get(sent.start_char, sent.text) for sent in doc.sents]
    return (" ".join(updated_text), entity_map) if return_map else " ".join(updated_text)

def map_legal_to_structured_text(text, return_entity_map=False):
    """
    Applies entity tagging and introduces structural section/indentation markers to legal text.
    """
    text = clean_long_quotes(text)
    tagged_text, entity_map = tag_entities_with_spacy(text, return_map=True) if return_entity_map else (tag_entities_with_spacy(text), None)
    text = tagged_text.replace("--", ":")

    section_counter = 1
    lines = text.split("\n")
    structured, current_path = [], []
    in_section = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if re.match(r'^(SEC\\.|SECTION)\\s+\\d+\\.', stripped, flags=re.IGNORECASE):
            if in_section:
                structured.append(f"[SEC{section_counter}_END]")
                section_counter += 1
            structured.append(f"[SEC{section_counter}_START]")
            current_path = [str(section_counter)]
            in_section = True
            structured.append(stripped)
            continue

        # Nested sub-structure detection
        if re.match(r"^\\([a-z]\\)", stripped):
            current_path = current_path[:1] + [stripped[1]]
        elif re.match(r"^\\(\\d+\\)", stripped):
            num = re.findall(r"\\((\\d+)\\)", stripped)[0]
            current_path = current_path[:2] + [num]
        elif re.match(r"^\\([A-Z]\\)", stripped):
            current_path = current_path[:3] + [stripped[1]]
        elif re.match(r"^\\([ivxlcdm]+\\)", stripped, flags=re.IGNORECASE):
            roman = re.findall(r"\\((.*?)\\)", stripped)[0]
            current_path = current_path[:4] + [roman.lower()]

        path_str = ".".join(current_path)
        path_prefix = f"[PATH={path_str}]"
        indent = "  " * len(current_path)
        structured.append(f"{indent}{path_prefix} {stripped}")

    if in_section:
        structured.append(f"[SEC{section_counter}_END]")

    output = "\n".join(structured)
    output = re.sub(r'\s+([.,;:])', r'\1', output)
    return (output, entity_map) if return_entity_map else output

# ======== ECP SCORING & TAGGING ========

def extract_entities(text):
    """
    Extracts all canonical entity placeholders from the text.
    """
    return set(re.findall(r'@[\w]+_\d+@', text))

def compute_ecp(input_text, summary_text):
    """
    Computes entity coverage precision (ECP) — fraction of input entities present in the summary.
    """
    input_entities = extract_entities(input_text)
    matched = [e for e in input_entities if e.lower() in summary_text.lower()]
    return 1.0 if not input_entities else len(matched) / len(input_entities)

def get_ecp_tag(ecp):
    """
    Returns a control tag based on ECP score thresholds.
    """
    if ecp >= 0.4:
        return "[ECP=HIGH]"
    elif ecp >= 0.25:
        return "[ECP=MEDIUM]"
    else:
        return "[ECP=LOW]"

def get_section_tag(text):
    """
    Tags the document as having many or few sections based on [SEC]_ markers.
    """
    count = len(re.findall(r"\[SEC\\d+_START\]", text))
    return "[SEC=MANY]" if count >= 5 else "[SEC=FEW]"

def extract_summary_terms(summary):
    """
    Extracts named entities and meaningful noun chunks from the summary for scoring.
    """
    doc = nlp(summary)
    terms = set()
    for ent in doc.ents:
        if ent.label_ not in {"CARDINAL", "PERCENT", "ORDINAL"}:
            terms.add(ent.text.lower())
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower().strip()
        if len(chunk_text.split()) == 1 and chunk_text in nlp.Defaults.stop_words:
            continue
        terms.add(chunk_text)
    return terms

def score_sentences_by_summary_terms(input_text, summary_terms):
    """
    Scores each sentence in input by how many summary terms it contains.
    Skips boilerplate legal language.
    """
    sentences = nltk.sent_tokenize(input_text)
    sentence_scores = []
    for sent in sentences:
        sent_lower = sent.lower()
        if any(skip in sent_lower for skip in ["appropriated", "fiscal year", "authorized", "such sums", "treasury", "public law", "act of", "21 u.s.c."]):
            continue
        score = sum(1 for term in summary_terms if term in sent_lower)
        sentence_scores.append((score, sent))
    return sentence_scores

def compute_sentence_ecp(sentences, summary_text):
    """
    Computes ECP at sentence level for later tagging.
    """
    results = []
    for sent in sentences:
        ents = extract_entities(sent)
        if not ents:
            results.append((sent, None))
            continue
        matched = [e for e in ents if e.lower() in summary_text.lower()]
        ecp = len(matched) / len(ents)
        results.append((sent, ecp))
    return results

def tag_sentence_with_ecp(sent, ecp):
    """
    Adds a sentence-level ECP control tag to the sentence.
    """
    if ecp is None:
        return sent
    elif ecp >= 0.6:
        return f"[SENT_ECP=HIGH] {sent}"
    elif ecp >= 0.3:
        return f"[SENT_ECP=MEDIUM] {sent}"
    else:
        return f"[SENT_ECP=LOW] {sent}"

def tag_sentences_by_ecp(input_text, summary_text):
    """
    Applies sentence-level ECP tagging to the input.
    """
    sentences = nltk.sent_tokenize(input_text)
    scored = compute_sentence_ecp(sentences, summary_text)
    tagged = [tag_sentence_with_ecp(sent, ecp) for sent, ecp in scored]
    return " ".join(tagged)

# ======== CUSTOM TRAINER EXTENSIONS ========

class DataCollatorWithWeights(DataCollatorForSeq2Seq):
    """
    Extends DataCollator to support sample-level weights during training.
    """
    def __call__(self, features, return_tensors=None):
        if "sample_weights" in features[0]:
            weights = [f.pop("sample_weights") for f in features]
            batch = super().__call__(features, return_tensors=return_tensors)
            batch["sample_weights"] = torch.tensor(weights, dtype=torch.float)
        else:
            batch = super().__call__(features, return_tensors=return_tensors)
        return batch

class WeightedLossTrainer(Seq2SeqTrainer):
    """
    Custom Trainer that applies sample weights to the training loss.
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        weights = torch.tensor(inputs.pop("sample_weights", [1.0]*labels.size(0))).to(model.device)
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        mask = labels.view(-1) != -100
        loss = loss * mask
        loss = loss.view(labels.size(0), -1).sum(dim=1) / mask.view(labels.size(0), -1).sum(dim=1)
        weighted_loss = (loss * weights).mean()
        return (weighted_loss, outputs) if return_outputs else weighted_loss

class CurriculumCallback(TrainerCallback):
    """
    Switches to a larger/full dataset after a certain epoch — used for curriculum learning.
    """
    def __init__(self, trainer_ref, switch_epoch=6):
        self.trainer_ref = trainer_ref
        self.switch_epoch = switch_epoch

    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.epoch == self.switch_epoch:
            print(f"\n[Curriculum] Switching to full dataset at epoch {state.epoch}")
            self.trainer_ref.train_dataset = self.trainer_ref.train_dataset_full

class SummaCEarlyStoppingCallback(TrainerCallback):
    """
    Custom callback for early stopping based on SummaC consistency score.
    """
    def __init__(self, tokenizer, eval_dataset, patience=3, min_delta=0.5, eval_every_n_epochs=2):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.patience = patience
        self.min_delta = min_delta
        self.eval_every_n_epochs = eval_every_n_epochs
        self.best_score = -float('inf')
        self.counter = 0
        self.summac_model = SummaCConv(models=["mnli"], granularity="sentence", device="cuda" if torch.cuda.is_available() else "cpu")

    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch)
        if current_epoch % self.eval_every_n_epochs != 0:
            return control
        model = kwargs["model"]
        inputs = [ex["input"] for ex in self.eval_dataset]
        references = [ex["label"] for ex in self.eval_dataset]
        generated = []
        for i in range(0, len(inputs), 8):
            batch = inputs[i:i+8]
            tokenized = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **tokenized,
                    max_length=384,
                    num_beams=5,
                    no_repeat_ngram_size=3
                )
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated.extend(decoded)
        score = self.summac_model.score(references, generated, aggregation="mean")["scores"][0] * 100
        print(f"\n[SummaC] Epoch {current_epoch}: {score:.2f} (best so far: {self.best_score:.2f})")
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(f"[SummaC Early Stopping] No improvement. Counter = {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print("[SummaC Early Stopping] Triggered — stopping training.")
                control.should_training_stop = True
        return control
