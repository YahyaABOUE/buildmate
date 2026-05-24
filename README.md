# BuildMate 🖥️
### Multi-Agent AI PC Recommendation System

**BRIM** — Yahya ABOUELAZIZ · Mehdi GUELLIDA · Younes EL MAJDOUBI EL IDRISSI  
S8 Integrated Project · AI & Big Data Program · UIR 2025–2026  
Supervisor: Prof. Hakim Hafidi

---

## Overview

BuildMate is a multi-agent AI system that helps users find the most suitable PC build or laptop according to their needs, preferences, and budget. The system combines natural language understanding, deep learning, deterministic recommendation logic, and multi-agent orchestration to deliver explainable, validated hardware recommendations.

---

## Architecture

```
User Input (free text or structured)
        ↓
Language Check (English only)
        ↓
NLP Classification Agent (DistilBERT)
        ↓
Confidence Check (threshold filtering)
        ↓
Orchestrator — Override Logic
        ↓
Recommendation Agent → Hardware JSON Database
        ↓
Compatibility Agent → Deterministic Rules
        ↓
Human-in-the-Loop Checkpoint
        ↓
Final Recommendation
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | CrewAI |
| LLM Backend | Gemini API |
| Deep Learning | PyTorch 2.0 + DistilBERT |
| UI | Gradio + HTML/CSS/JS |
| Language | Python 3.11 |

---

## Project Structure

```
buildmate/
├── data/
│   ├── hardware/
│   │   └── components.json       # Hardware database
│   └── training/
│       └── dataset.json          # NLP training data
├── models/
│   ├── train_classifier.py       # DistilBERT training script
│   └── checkpoints/              # Saved model weights (see below)
├── agents/
│   ├── orchestrator.py           # Main pipeline coordinator
│   ├── nlp_agent.py              # NLP classification agent
│   ├── recommendation_agent.py   # Hardware recommendation agent
│   └── compatibility_agent.py    # Compatibility validation agent
├── tools/
│   ├── classifier_tool.py        # DistilBERT wrapped as CrewAI tool
│   ├── hardware_tool.py          # Hardware database query tool
│   └── compatibility_tool.py     # Deterministic compatibility rules
├── ui/
│   ├── app.py                    # Gradio backend + API
│   └── index.html                # Multi-page SPA frontend
├── logs/                         # JSON logs (auto-generated)
├── main.py                       # Terminal entry point
├── requirements.txt
└── .env                          # API keys (not committed)
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YahyaABOUE/buildmate.git
cd buildmate
```

### 2. Install dependencies

```bash
python3.11 -m pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the root directory:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com)

### 4. Download the trained model

Download `best_model.pt` from [Google Drive link] and place it in `models/checkpoints/`.

### 5. Train the model yourself (optional)

```bash
python3.11 models/train_classifier.py
```

Training takes ~5 minutes on Apple Silicon (MPS) or CPU.

### 6. Run the application

```bash
python3.11 ui/app.py
```

Opens the UI at `http://localhost:3000/index.html`  
Gradio API runs at `http://localhost:7860`

### 7. Terminal mode (optional)

```bash
python3.11 main.py
```

---

## Model Performance

Fine-tuned on 300 synthetic samples · 10 epochs · AdamW lr=2e-5 · PyTorch 2.0

| Task | Accuracy | Notes |
|---|---|---|
| Device Type | 92% | Desktop vs Laptop |
| Budget Tier | 89% | Low / Mid / High |
| Use Case | 72% | 6 classes |

---

## Agents

### 🤖 NLP Classification Agent
Uses fine-tuned DistilBERT to classify free-text into `use_case`, `device_type`, and `budget_tier` with confidence scores. Low confidence inputs are rejected.

### ⚙️ Recommendation Agent
Queries the hardware JSON database and selects optimal components or laptop. Every selection includes a detailed technical explanation.

### 🔍 Compatibility Agent
Runs deterministic hardware validation:
- CPU socket ↔ Motherboard socket
- RAM type (DDR4/DDR5) ↔ CPU platform and Motherboard
- PSU wattage ≥ CPU TDP + GPU TDP + 100W overhead
- RAM capacity ≤ Motherboard maximum
- Total price ≤ User budget

### 🧠 Orchestrator
Coordinates the full pipeline. Enforces override hierarchy (explicit user selections always win over NLP predictions), manages retries, logs all actions, and manages the HITL checkpoint.

---

## Hardware Database

| Category | Count | Coverage |
|---|---|---|
| CPUs | 7 | Intel LGA1700 · AMD AM4/AM5 |
| GPUs | 8 | NVIDIA RTX 3060–4090 · AMD RX series |
| Motherboards | 7 | DDR4 & DDR5 · 3 sockets |
| RAM | 4 | DDR4/DDR5 · 16–64GB |
| Storage | 4 | NVMe 1–2TB · HDD |
| PSUs | 4 | 550W–1000W · 80+ rated |
| Cases | 4 | Micro-ATX & ATX |
| Laptops | 8 | All tiers and use cases |

---

## Error Handling

| Input | Behavior |
|---|---|
| Gibberish | Rejected — too ambiguous |
| French / Arabic | Rejected — English only |
| Too vague | Rejected — confidence below threshold |
| Contradictory | Low confidence warning, resolved intelligently |
| Budget too low | Builds cheapest viable configuration |
| Compatibility failure | Automatic retry with relaxed budget (3 attempts) |

---

## Logging

Every agent action is logged with ISO timestamps to `logs/` in JSON format:

```json
{
  "timestamp": "2026-05-24T04:09:20.053410",
  "agent": "nlp_agent",
  "action": "classification_complete",
  "details": {
    "use_case": {"label": "gaming", "confidence": 0.88},
    "device_type": {"label": "laptop", "confidence": 0.82},
    "budget_tier": {"label": "low", "confidence": 0.43}
  }
}
```

---

## Academic Integrity

LLM assistants were used as tools during development. All code was reviewed, understood, and can be explained in full during the oral defense.

---

## License

Academic project — UIR 2025–2026. Not for commercial use.
