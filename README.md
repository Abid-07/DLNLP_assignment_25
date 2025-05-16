# DLNLP_assignment_25

# Legal Summarizer – Entity- and Structure-Aware Summarization

This project implements a legal document summarizer using a fine-tuned version of the [Salesforce/CodeT5-small](https://huggingface.co/Salesforce/codet5-small) model. It integrates structured and entity-aware preprocessing with ROUGE and SummaC-based evaluation for factual and linguistic quality.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/DLNLP_assignment_25.git
cd DLNLP_assignment_25
```

### 2. Create & Activate Virtual Environment

Make sure you are **not using Python 3.13**, as some packages are not yet compatible. Python 3.11.x is recommended.

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Requirements

Install all required libraries (including Transformers, Datasets, SummaC, etc.):

```bash
pip install -r requirements.txt
```

> If installation fails due to version conflicts, ensure you are using Python 3.11 and not a newer unsupported version.

---

##  Download Model Weights

Due to file size restrictions, model weights are **not included in the repository**.

Download the trained model and tokenizer from this Google Drive link:
**[ Download Final Model (Google Drive)](https://drive.google.com/drive/folders/13XCstmkZpnhfSkVbC4iV20H9wQjcmZbp?dmr=1&ec=wgc-drive-hero-goto)**

Once downloaded, extract and place the folder at:

```bash
./codet5-dual-summary/final_model/
```

---

## 💠 Notes on spaCy

spaCy's English model must be installed manually. Run the following once in your environment:

```bash
python -m spacy download en_core_web_sm
```

This is **not** automatically handled by `requirements.txt`.

---

##  How to Run

### Training + Evaluation:

```bash
python main.py
```

* Trains the model on a preprocessed version of the [BillSum dataset](https://huggingface.co/datasets/billsum).
* Evaluates on a subset using both ROUGE and [SummaC](https://github.com/tingofurro/summac).

---

##  Hardware Requirements

* A **GPU** is strongly recommended, especially for training.
* The original training was done on Google Colab with an **L4 GPU**.
* Running on CPU is possible but **extremely slow** for both training and evaluation.

---

##  Evaluation Metrics

* **ROUGE-L**: Measures lexical overlap between generated and reference summaries.
* **SummaC (MNLI)**: Sentence-level entailment-based factual consistency.

---

##  Project Structure

```
DLNLP_assignment_25/
│
├── A/
│   ├── model.py          # Model wrapper and trainer
│   ├── data.py           # Dataset preprocessing and tokenization
│   └── utils.py          # Utility functions and custom Trainer logic
│
├── main.py               # Entry point: training and evaluation
├── requirements.txt      # Package requirements
└── README.md             # You are here
```

---

