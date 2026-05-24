import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch import nn
import warnings
warnings.filterwarnings("ignore")

DATASET_PATH = "data/training/dataset.json"
CHECKPOINT_DIR = "models/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LABEL_MAPS = {
    "use_case": {"gaming": 0, "video_editing": 1, "machine_learning": 2, "office": 3, "3d_rendering": 4, "music_production": 5},
    "device_type": {"desktop": 0, "laptop": 1},
    "budget_tier": {"low": 0, "mid": 1, "high": 2}
}

REVERSE_MAPS = {
    task: {v: k for k, v in mapping.items()}
    for task, mapping in LABEL_MAPS.items()
}

class BuildMateDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels["use_case"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "use_case": torch.tensor(self.labels["use_case"][idx]),
            "device_type": torch.tensor(self.labels["device_type"][idx]),
            "budget_tier": torch.tensor(self.labels["budget_tier"][idx])
        }

class BuildMateClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden = self.bert.config.hidden_size

        self.dropout = nn.Dropout(0.3)
        self.use_case_head = nn.Linear(hidden, 6)
        self.device_type_head = nn.Linear(hidden, 2)
        self.budget_tier_head = nn.Linear(hidden, 3)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(output.last_hidden_state[:, 0, :])
        return {
            "use_case": self.use_case_head(cls),
            "device_type": self.device_type_head(cls),
            "budget_tier": self.budget_tier_head(cls)
        }

def load_data():
    with open(DATASET_PATH, "r") as f:
        data = json.load(f)

    texts = [d["text"] for d in data]
    labels = {
        "use_case": [LABEL_MAPS["use_case"][d["use_case"]] for d in data],
        "device_type": [LABEL_MAPS["device_type"][d["device_type"]] for d in data],
        "budget_tier": [LABEL_MAPS["budget_tier"][d["budget_tier"]] for d in data]
    }
    return texts, labels

def train():
    print("Loading dataset...")
    texts, labels = load_data()

    indices = list(range(len(texts)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    train_labels = {k: [v[i] for i in train_idx] for k, v in labels.items()}
    val_labels = {k: [v[i] for i in val_idx] for k, v in labels.items()}

    print("Loading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    train_dataset = BuildMateDataset(train_texts, train_labels, tokenizer)
    val_dataset = BuildMateDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BuildMateClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 10
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)

            loss = (
                criterion(outputs["use_case"], batch["use_case"].to(device)) +
                criterion(outputs["device_type"], batch["device_type"].to(device)) +
                criterion(outputs["budget_tier"], batch["budget_tier"].to(device))
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        all_preds = {"use_case": [], "device_type": [], "budget_tier": []}
        all_labels = {"use_case": [], "device_type": [], "budget_tier": []}

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids, attention_mask)

                loss = (
                    criterion(outputs["use_case"], batch["use_case"].to(device)) +
                    criterion(outputs["device_type"], batch["device_type"].to(device)) +
                    criterion(outputs["budget_tier"], batch["budget_tier"].to(device))
                )
                val_loss += loss.item()

                for task in all_preds:
                    preds = torch.argmax(outputs[task], dim=1).cpu().numpy()
                    all_preds[task].extend(preds)
                    all_labels[task].extend(batch[task].numpy())

        avg_val_loss = val_loss / len(val_loader)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        for task in all_preds:
            acc = accuracy_score(all_labels[task], all_preds[task])
            print(f"  {task} accuracy: {acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pt")
            print("  Saved best model")

    print("\nTraining complete. Generating final evaluation...")
    model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/best_model.pt"))
    model.eval()

    all_preds = {"use_case": [], "device_type": [], "budget_tier": []}
    all_labels = {"use_case": [], "device_type": [], "budget_tier": []}

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)
            for task in all_preds:
                preds = torch.argmax(outputs[task], dim=1).cpu().numpy()
                all_preds[task].extend(preds)
                all_labels[task].extend(batch[task].numpy())

    for task in all_preds:
        print(f"\n--- {task.upper()} ---")
        print(classification_report(
            all_labels[task], all_preds[task],
            target_names=list(LABEL_MAPS[task].keys())
        ))
        print("Confusion Matrix:")
        print(confusion_matrix(all_labels[task], all_preds[task]))

    tokenizer.save_pretrained(CHECKPOINT_DIR)
    print(f"\nTokenizer saved to {CHECKPOINT_DIR}")
    print("Done.")

if __name__ == "__main__":
    train()