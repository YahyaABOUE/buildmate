import torch
import json
import os
from transformers import DistilBertTokenizer, DistilBertModel
from torch import nn
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

CHECKPOINT_DIR = "models/checkpoints"

LABEL_MAPS = {
    "use_case": {"gaming": 0, "video_editing": 1, "machine_learning": 2, "office": 3, "3d_rendering": 4, "music_production": 5},
    "device_type": {"desktop": 0, "laptop": 1},
    "budget_tier": {"low": 0, "mid": 1, "high": 2}
}

REVERSE_MAPS = {
    task: {v: k for k, v in mapping.items()}
    for task, mapping in LABEL_MAPS.items()
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

_model = None
_tokenizer = None

def load_model():
    global _model, _tokenizer
    if _model is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        _tokenizer = DistilBertTokenizer.from_pretrained(CHECKPOINT_DIR)
        _model = BuildMateClassifier()
        _model.load_state_dict(torch.load(
            f"{CHECKPOINT_DIR}/best_model.pt",
            map_location=device
        ))
        _model.to(device)
        _model.eval()
    return _model, _tokenizer

class ClassifierInput(BaseModel):
    text: str = Field(description="The user's free-text request to classify")

class ClassifierTool(BaseTool):
    name: str = "nlp_classifier"
    description: str = (
        "Classifies a user's free-text PC request into use_case, device_type, and budget_tier. "
        "Input is the raw user text. Returns a JSON string with predicted labels and confidence scores."
    )
    args_schema: type[BaseModel] = ClassifierInput

    def _run(self, text: str) -> str:
        try:
            model, tokenizer = load_model()
            device = next(model.parameters()).device

            encoding = tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            )

            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask)

            results = {}
            for task, logits in outputs.items():
                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                confidence = probs[pred_idx].item()
                results[task] = {
                    "label": REVERSE_MAPS[task][pred_idx],
                    "confidence": round(confidence, 4)
                }

            return json.dumps(results)

        except Exception as e:
            return json.dumps({"error": str(e)})