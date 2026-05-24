**`README.md`**

```markdown
# BuildMate 
### Multi-Agent AI PC Recommendation System
**BRIM** — Yahya ABOUELAZIZ · Mehdi GUELLIDA · Younes EL MAJDOUBI EL IDRISSI  
S8 Integrated Project · AI & Big Data Program · UIR 2025–2026  
Supervisor: Prof. Hakim Hafidi

---

## Overview

BuildMate is a multi-agent AI system that helps users find the most suitable PC build or laptop according to their needs, preferences, and budget.

The system combines:
- **Natural language understanding** via a fine-tuned DistilBERT model
- **Multi-agent orchestration** via CrewAI
- **Deterministic compatibility checking** via hardware rules
- **Human-in-the-loop** approval checkpoint

---

## Architecture

```
User Input (free text or structured)
        ↓
NLP Classification Agent (DistilBERT)
        ↓
Orchestrator (CrewAI)
        ↓
Recommendation Agent → Hardware JSON Database
        ↓
Compatibility Agent → Deterministic Rules
        ↓
Human-in-the-Loop Checkpoint
        ↓
Final Recommendation
```

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
│   └── checkpoints/              # Saved model weights
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
git clone https://github.com/your-repo/buildmate.git
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

### 4. Train the DistilBERT model

```bash
python3.11 models/train_classifier.py
```

Training takes ~5 minutes on Apple Silicon (MPS) or CPU.  
The trained model will be saved to `models/checkpoints/best_model.pt`.

### 5. Run the application

```bash
python3.11 ui/app.py
```

This will open the UI at `http://localhost:3000/index.html`  
The Gradio API runs at `http://localhost:7860`

### 6. (Optional) Terminal mode

```bash
python3.11 main.py
```

---

## Model Performance

The DistilBERT classifier was fine-tuned on 300 synthetic PC request samples for 10 epochs.

| Task | Accuracy |
|---|---|
| Use Case Classification | 72% |
| Device Type Classification | 92% |
| Budget Tier Classification | 89% |

---

## Agents

### 1. NLP Classification Agent
Uses a fine-tuned DistilBERT model to classify user free-text input into:
- `use_case`: gaming, video_editing, machine_learning, office, 3d_rendering, music_production
- `device_type`: desktop, laptop
- `budget_tier`: low, mid, high

### 2. Recommendation Agent
Queries the hardware JSON database and selects the best matching components or laptop based on use case, device type, and budget tier.

### 3. Compatibility Agent
Runs deterministic hardware compatibility checks:
- CPU socket vs motherboard socket
- RAM type compatibility
- PSU wattage vs estimated TDP
- Budget validation

### 4. Orchestrator
Coordinates all agents, manages retries, detects low confidence, handles errors, logs every action, and manages the HITL checkpoint.

---

## Hardware Database

The system uses a curated JSON database of real components:
- 7 CPUs (Intel & AMD, DDR4/DDR5)
- 7 GPUs (NVIDIA & AMD, low/mid/high tier)
- 6 Motherboards (LGA1700 & AM4/AM5)
- 4 RAM kits (DDR4 & DDR5)
- 4 Storage options (NVMe & HDD)
- 4 PSUs (550W–1000W)
- 4 Cases
- 8 Laptops

---

## Logging

Every agent action is logged with timestamps to `logs/` in JSON format:

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

## Error Handling

- **Gibberish input** → rejected with clear error message
- **French/Arabic input** → rejected, English only
- **Too vague input** → rejected if average NLP confidence < 55%
- **Compatibility failure** → automatic retry with relaxed budget (up to 3 attempts)
- **Missing inputs** → validated before pipeline runs

---

## Academic Integrity

LLM assistants were used as tools during development. All code was reviewed, understood, and can be explained in full during the oral defense.

---

## License

Academic project — UIR 2025–2026. Not for commercial use.
```