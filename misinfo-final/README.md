# 🛡️ AI Against Misinformation
### CodeWizards 2.0 | SRMIST Delhi-NCR | 17–18 April 2026

> Real-time · Multimodal · Multilingual · Explainable · Active countering

---

## What This Does

Detects, flags, and **actively counters** misinformation across text, images, audio, and video — in English, Hindi, and Tamil — in real time.

Unlike passive detection tools, this system generates structured **counter-narratives** in the user's own language, predicts **virality**, maps **narrative clusters**, and serves results from a **Qdrant semantic cache** for sub-100ms repeat lookups.

---

## Quick Start

```bash
# 1. Clone / unzip project
cd misinformation-ai

# 2. One-shot setup
bash scripts/setup.sh

# 3. Start the API
cd backend
uvicorn app.main:app --reload --port 8000

# 4. Open docs
open http://localhost:8000/docs
```

**Or manually:**

```bash
cd backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                               # Fill in API keys
python ../qdrant/init_collections.py               # Create Qdrant collections
python ../qdrant/seed_data.py                      # Seed 14 known claims
uvicorn app.main:app --reload --port 8000
```

---

## API Keys Required

| Key | Where to get | Required? |
|---|---|---|
| `QDRANT_URL` + `QDRANT_API_KEY` | [cloud.qdrant.io](https://cloud.qdrant.io) (free) | ✅ Yes |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | ⚪ Optional (rule-based fallback) |
| `GOOGLE_FACT_CHECK_API_KEY` | [Google Cloud Console](https://developers.google.com/fact-check/tools/api) (free) | ⚪ Optional (NLI fallback) |
| `SERPER_API_KEY` | [serper.dev](https://serper.dev) (100 free/mo) | ⚪ Optional |

**The system works without any paid keys** — Qdrant's free tier + open-source models cover 100% of functionality.

---

## Feature Overview

### Main Problem Statement Features

| Feature | Endpoint | Description |
|---|---|---|
| Text analysis | `POST /api/v1/analyze` | Full pipeline: detect → verify → score → counter |
| URL analysis | `POST /api/v1/analyze/url` | Fetch article, then full pipeline |
| Batch analysis | `POST /api/v1/analyze/batch` | Up to 5 claims at once |
| Image deepfake | `POST /api/v1/analyze/image` | EfficientNet-B4 + caption fact-check |
| Audio fact-check | `POST /api/v1/analyze/audio` | Whisper ASR → full pipeline |
| Video analysis | `POST /api/v1/analyze/video` | Frame sampling + audio transcription |
| Direct fact-check | `POST /api/v1/fact-check` | Multi-source verification only |
| Counter-narrative | `POST /api/v1/counter` | Claude-powered, multilingual |
| Hindi support | All endpoints | Auto-detect + translate + respond in Hindi |
| Tamil support | All endpoints | Auto-detect + translate + respond in Tamil |

### Unique Features (Never Combined Before)

| Feature | Endpoint | Description |
|---|---|---|
| 🔥 Trending tracker | `GET /api/v1/trending` | Most-checked misinfo ranked by Qdrant hit count |
| 🚀 Virality prediction | `POST /api/v1/predict/virality` | Predict spread based on Qdrant cluster size |
| 🕸️ Narrative network | `POST /api/v1/network` | Map coordinated disinformation clusters |
| 🔍 Explainability | In every response | LIME-style feature attribution per risk score |
| ⚡ Semantic cache | Automatic | Qdrant dedup — same claim = instant result |

### Special Track (Duality AI Segmentation)

| Feature | Endpoint | Description |
|---|---|---|
| Class list | `GET /api/v1/segmentation/classes` | 10 desert environment classes |
| IoU analysis | `POST /api/v1/segmentation/analyze` | Per-class IoU with failure detection |
| Class similarity | `GET /api/v1/segmentation/similar-classes` | Qdrant-powered misclassification analysis |
| Report template | `GET /api/v1/segmentation/report-template` | Structured Duality AI report format |

---

## Project Structure

```
misinformation-ai/
│
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app, all routers
│   │   ├── config.py                # Settings from .env
│   │   ├── models/
│   │   │   ├── enums.py             # Verdict, Language, ContentType, RiskLevel
│   │   │   └── schemas.py           # 15+ Pydantic request/response models
│   │   ├── routers/
│   │   │   ├── analyze.py           # Core text/URL/batch analysis
│   │   │   ├── fact_check.py        # Direct fact verification
│   │   │   ├── counter.py           # Counter-narrative generation
│   │   │   ├── multimodal.py        # Image / audio / video
│   │   │   ├── trending.py          # Trending, virality, network, stats
│   │   │   └── segmentation.py      # Duality AI special track
│   │   └── services/
│   │       ├── preprocessor.py      # Text cleaning, URL fetching
│   │       ├── translator.py        # Hindi/Tamil detection & translation
│   │       ├── claim_extractor.py   # Claim extraction & classification
│   │       ├── fact_verifier.py     # 4-source ensemble fact-check
│   │       ├── detector.py          # Deepfake, AI-text, credibility
│   │       ├── scorer.py            # Risk scoring + explainability
│   │       ├── counter_gen.py       # Counter-narrative (Claude + fallback)
│   │       └── qdrant_service.py    # All Qdrant operations
│   ├── .env                         # Your API keys (not committed)
│   ├── .env.example                 # Template
│   └── requirements.txt
│
├── ml/
│   ├── deepfake/
│   │   ├── model.py                 # EfficientNet-B4 wrapper
│   │   └── inference.py             # CLI: python inference.py image.jpg
│   ├── ai_detect/
│   │   ├── model.py                 # RoBERTa detector + heuristic fallback
│   │   └── inference.py             # CLI: python inference.py "text"
│   └── embeddings/
│       └── embed.py                 # Multilingual embedding utility
│
├── qdrant/
│   ├── init_collections.py          # Create all 4 collections
│   └── seed_data.py                 # Pre-seed 14 known claims
│
├── scripts/
│   ├── setup.sh                     # One-shot setup script
│   └── demo_data/
│       └── sample_requests.json     # All demo requests ready to fire
│
├── docs/
│   ├── ARCHITECTURE.md              # Full system architecture
│   ├── API.md                       # Complete API reference
│   └── DEMO_SCRIPT.md               # Judge presentation guide
│
└── README.md                        # This file
```

---

## How the Fact-Check Pipeline Works

```
Claim → Qdrant cache? ──YES──► Return cached result (< 100ms)
              │
              NO
              ▼
    Google Fact Check API
    (authoritative — returns immediately if found)
              │
              ▼ (if no result)
    Wikipedia fetch + BART-MNLI entailment (contradiction score)
              │
              ▼ (combined)
    Serper web search + BART-MNLI entailment
              │
              ▼ (combined)
    Standalone NLI: true / false / misleading
              │
              ▼
    Weighted ensemble → FAKE / MISLEADING / TRUE / UNVERIFIED
              │
              ▼
    Risk score (0-100) + LIME explainability
              │
              ▼
    Counter-narrative (Claude / Hindi templates)
              │
              ▼
    Upsert to Qdrant → future requests cached
```

---

## Risk Score Weights

| Signal | Weight | Description |
|---|---|---|
| Fact-check result | **40%** | Most important — authoritative verification |
| AI-generated prob | 20% | RoBERTa detector |
| Deepfake prob | 15% | EfficientNet-B4 |
| Source credibility | 15% | Domain ranking (BBC=0.95, unknown=0.50) |
| NLI signals | 10% | BART-large-MNLI raw scores |

---

## Environment

Tested on Python 3.11. CPU-only — no GPU required.  
All models run on CPU with acceptable latency for demo (2-5s for full pipeline, <100ms for cached).

---

## Team

Built at CodeWizards 2.0 Hackathon, SRMIST Delhi-NCR Campus, Ghaziabad.  
Special problem statement powered by Duality AI.
