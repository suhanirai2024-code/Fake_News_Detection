"""
Fine-tunes DistilBERT (a smaller, faster BERT variant) on the cleaned
fake/real news dataset produced by prepare_data.py.

Run AFTER prepare_data.py has generated dataset/cleaned_data.csv.

Requirements: torch, transformers  (pip install torch transformers)

NOTE ON COMPUTE:
This will be very slow on a CPU-only laptop (potentially several hours
for the full 44k-row dataset). Two options:
  1. Run this on Google Colab with a free GPU (Runtime > Change runtime
     type > GPU), uploading dataset/cleaned_data.csv there.
  2. Reduce dataset size for a quick local demo by setting SAMPLE_SIZE
     below to e.g. 3000.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5
OUTPUT_DIR = "bert_model"

# Set to an integer (e.g. 3000) to subsample for a fast local test run.
# Set to None to use the full cleaned dataset.
SAMPLE_SIZE = 500
PRINT_EVERY = 10  # how often (in steps) to print progress


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def main():
    print("Loading cleaned dataset...")
    data = pd.read_csv("dataset/cleaned_data.csv")

    # Combine title + text into one column if both exist
    if "title" in data.columns and "text" in data.columns:
        data["content"] = data["title"].fillna("") + " " + data["text"].fillna("")
    else:
        # fall back to first non-label column
        text_col = [c for c in data.columns if c != "label"][0]
        data["content"] = data[text_col].astype(str)

    if SAMPLE_SIZE:
        data = data.sample(n=min(SAMPLE_SIZE, len(data)), random_state=42)

    X = data["content"].tolist()
    y = data["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("WARNING: no GPU detected — training will be slow. See the "
              "note at the top of this file about Colab or SAMPLE_SIZE.")

    print(f"Downloading tokenizer/model: {MODEL_NAME} ...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    model.to(device)

    train_dataset = NewsDataset(X_train, y_train, tokenizer, MAX_LEN)
    test_dataset = NewsDataset(X_test, y_test, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if step % PRINT_EVERY == 0:
                print(f"  epoch {epoch+1} step {step}/{len(train_loader)} "
                      f"loss {loss.item():.4f}")

        print(f"Epoch {epoch+1}/{EPOCHS} — avg loss: {total_loss/len(train_loader):.4f}")

    # ---------------- Evaluation ----------------
    print("\nEvaluating on test split...")
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            preds.extend(batch_preds)
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    print(f"\nTest Accuracy: {acc:.4f}\n")
    print(classification_report(true_labels, preds, target_names=["Fake", "Real"]))

    print(f"\nSaving model + tokenizer to ./{OUTPUT_DIR}/ ...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done. app.py will now detect this folder and offer BERT as a model option.")


if __name__ == "__main__":
    main()