# main.py

from A.model import LegalSummarizer
from A.data import get_datasets
import torch
import os

# ===============================
# Data preprocessing
train_dataset, val_dataset, test_dataset = get_datasets()

# ===============================
# Task A: Legal Text Summarization
print("[INFO] Initializing model...")
model_A = LegalSummarizer()

print("[INFO] Training model...")
acc_A_train = model_A.train(train_dataset, val_dataset)

print("[INFO] Evaluating model...")
acc_A_test = model_A.test(test_dataset)

# Clean up memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ===============================
# Print Results
print("\n==============================")
print("FINAL RESULTS")
print('TA:{},{};'.format(acc_A_train, acc_A_test))
print("==============================")
