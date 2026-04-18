# System Architecture — AI Against Misinformation

## Overview

A real-time, multimodal, multilingual misinformation detection and active countering platform built for the CodeWizards 2.0 hackathon. The system goes beyond passive detection — it generates verified counter-narratives, predicts virality, and maps narrative clusters using Qdrant vector search.

---

## High-Level Pipeline

```
Input (Text / Image / Audio / Video / URL)
         │
         ▼
┌─────────────────────────────────────────┐
│  Layer 1: Preprocessing                 │
│  - HTML decode, URL strip, normalise    │
│  - Language detection (hi/ta/en)        │
│  - Google Translate (hi/ta → en)        │
│  - Whisper ASR (audio/video → text)     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Layer 2: Claim Extraction              │
│  - Sentence scoring heuristic           │
│  - Named entity recognition             │
│  - Claim type classification            │
│    (health / political / economic / …)  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Layer 3: Qdrant Cache Lookup           │
│  - multilingual-e5-large embedding      │
│  - Cosine similarity search             │
│  - Threshold: 0.85                      │
│  → HIT: return cached verdict (<100ms)  │
│  → MISS: proceed to verification        │
└──────────────┬──────────────────────────┘
               │ (cache miss only)
               ▼
┌─────────────────────────────────────────┐
│  Layer 4: Multi-source Fact Verification│
│                                         │
│  Priority 1: Google Fact Check API      │
│    → Authoritative (AFP, Snopes, BOOM)  │
│    → Returns immediately if found       │
│                                         │
│  Priority 2: Wikipedia + NLI            │
│    → Fetch article extract              │
│    → BART-large-MNLI entailment         │
│                                         │
│  Priority 3: Serper (Google Search)     │
│    → Top 3 web snippets                 │
│    → BART-large-MNLI entailment         │
│                                         │
│  Priority 4: Pure NLI Classification    │
│    → Standalone BART-large-MNLI         │
│    → Zero-shot: true/false/misleading   │
│                                         │
│  Ensemble: weighted vote → verdict      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Layer 5: Detection Signals             │
│  - AI text: RoBERTa-base-openai-detect  │
│  - Deepfake: EfficientNet-B4 (timm)     │
│  - Source credibility: domain ranking   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Layer 6: Risk Scoring + Explainability │
│                                         │
│  Weighted ensemble (sums to 1.0):       │
│    fact_check_result   → 0.40           │
│    ai_generated_prob   → 0.20           │
│    deepfake_prob        → 0.15          │
│    source_credibility  → 0.15           │
│    nli_signals         → 0.10           │
│                                         │
│  Score 0-100 → LOW/MEDIUM/HIGH/CRITICAL │
│  LIME-style feature attribution         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Layer 7: Active Countering (THE TWIST) │
│  - Claude API (primary)                 │
│  - Rule-based Hindi/Tamil templates     │
│    (fallback — no API key needed)       │
│  - Returns:                             │
│    • trusted summary                    │
│    • specific inconsistencies           │
│    • verified alternative               │
│    • actionable advice                  │
│    • citations                          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Layer 8: Qdrant Upsert + Response      │
│  - Store verified claim for future hits │
│  - Increment hit_count for trending     │
│  - Return full AnalysisResponse         │
└─────────────────────────────────────────┘
```

---

## Qdrant Collections

| Collection   | Purpose                        | Embedding Model                        | Dim |
|--------------|--------------------------------|----------------------------------------|-----|
| `claims`     | Verified claim cache           | paraphrase-multilingual-MiniLM-L12-v2  | 384 |
| `verdicts`   | Detailed verdict metadata      | paraphrase-multilingual-MiniLM-L12-v2  | 384 |
| `sources`    | Indexed trusted news articles  | paraphrase-multilingual-MiniLM-L12-v2  | 384 |
| `trending`   | High-frequency misinfo tracker | paraphrase-multilingual-MiniLM-L12-v2  | 384 |
| `seg_classes`| Segmentation class embeddings  | paraphrase-multilingual-MiniLM-L12-v2  | 384 |

### Why Qdrant is Critical

1. **Speed**: Cache HIT returns in <100ms vs 2-5s for full verification
2. **Multilingual dedup**: Hindi and English versions of the same claim map to similar vectors — one lookup covers both
3. **Trending tracker**: `hit_count` payload field incremented on every cache hit → powers the `/trending` endpoint
4. **Narrative clustering**: `/network` endpoint reveals coordinated disinformation campaigns by finding semantically similar claims
5. **Virality prediction**: Cluster size in Qdrant is a leading indicator of spread velocity
6. **Segmentation cross-use**: `seg_classes` collection stores CLIP-style class embeddings for misclassification analysis

---

## API Surface

```
/api/v1/
├── analyze          POST  Full text pipeline
├── analyze/url      POST  URL → extract → pipeline
├── analyze/batch    POST  Up to 5 texts at once
├── fact-check       POST  Direct fact verification
├── counter          POST  Counter-narrative only
├── analyze/image    POST  Image + caption
├── analyze/audio    POST  Audio → Whisper → pipeline
├── analyze/video    POST  Video frames + audio
├── detect/deepfake  POST  Quick deepfake probability
├── detect/ai-text   POST  Quick AI-text probability
├── trending         GET   Top misinfo by hit count  ★ UNIQUE
├── predict/virality POST  Virality prediction        ★ UNIQUE
├── network          POST  Narrative cluster map      ★ UNIQUE
├── stats            GET   System statistics
└── segmentation/
    ├── classes      GET   Duality AI class list
    ├── analyze      POST  IoU computation
    ├── similar-classes GET Qdrant class similarity
    └── report-template GET Report structure
```

---

## ML Models Used

| Model | Task | Size | Load |
|---|---|---|---|
| `facebook/bart-large-mnli` | NLI / Fact classification | 1.6GB | Lazy, once |
| `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual embeddings | 470MB | Lazy, once |
| `roberta-base-openai-detector` | AI text detection | 500MB | Lazy, once |
| `efficientnet_b4` (timm) | Deepfake image detection | 74MB | Lazy, once |
| `openai/whisper-base` | Audio transcription | 145MB | Lazy, once |

All models use **lazy loading** — they are loaded on first request, not at startup. This keeps cold-start fast.

---

## Unique Features (Never-Before-Built Combination)

### 1. Multilingual Semantic Claim Deduplication
Same misinformation in Hindi, Tamil, and English maps to the same vector cluster — one verification covers all three. No existing fact-checking system does this with Qdrant.

### 2. Virality Prediction Engine
Uses Qdrant cluster size + verdict + risk score to predict how fast and far a claim will spread. Gives authorities early warning before a piece of misinformation goes viral.

### 3. Narrative Network Analysis
Maps semantic clusters of claims in Qdrant to reveal coordinated disinformation campaigns. Shows whether a claim is isolated or part of a broader narrative family.

### 4. LIME-style Explainability
Every risk score comes with a feature attribution breakdown — not just "this is fake" but exactly which signals (deepfake, AI-generated, NLI, source credibility) drove the score.

### 5. Active Countering in User's Language
The system doesn't just flag — it generates a full counter-narrative in the user's own language (Hindi/Tamil/English) with specific inconsistency callouts and verified citations.

### 6. Segmentation + Qdrant Cross-Integration
The special track segmentation classes are embedded into Qdrant, enabling semantic misclassification analysis — finding which desert environment classes are most likely to be confused by the model.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Web Framework | FastAPI 0.115 + Uvicorn |
| Vector DB | Qdrant Cloud (free tier) |
| Embeddings | sentence-transformers (multilingual) |
| NLI | BART-large-MNLI (HuggingFace) |
| Deepfake | EfficientNet-B4 (timm, ImageNet pretrained) |
| AI Detect | RoBERTa-base-openai-detector |
| ASR | OpenAI Whisper base |
| LLM Counter | Anthropic Claude Sonnet 4 |
| Fact Check | Google Fact Check Tools API |
| Web Search | Serper (Google Search API) |
| Translation | Google Translate (free endpoint) |
| Language Detect | langdetect + Unicode range rules |
| Validation | Pydantic v2 |
| Logging | Loguru |
