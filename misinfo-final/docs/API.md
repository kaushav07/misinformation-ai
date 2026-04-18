# API Reference — AI Against Misinformation

Base URL: `http://localhost:8000`  
API Prefix: `/api/v1`  
Docs UI: `http://localhost:8000/docs`

---

## System

### `GET /health`
Returns service status and model configuration.

```json
{
  "status": "ok",
  "version": "2.0.0",
  "services": {
    "qdrant": "connected",
    "nli_model": "facebook/bart-large-mnli",
    "claude": "configured"
  }
}
```

---

## Analysis

### `POST /api/v1/analyze`
Full pipeline — text input. Supports English, Hindi, Tamil.

**Request**
```json
{
  "text": "5G towers spread COVID-19 through electromagnetic radiation.",
  "target_language": "en",
  "include_counter": true
}
```

**Response**
```json
{
  "original_text": "5G towers spread COVID-19...",
  "detected_language": "en",
  "translated_text": null,
  "content_type": "text",
  "claim": {
    "claim": "5G towers spread COVID-19 through electromagnetic radiation.",
    "claim_type": "health",
    "entities": ["COVID-19"]
  },
  "fact_check": {
    "verdict": "FAKE",
    "confidence": 0.94,
    "explanation": "...",
    "sources": ["WHO", "Reuters"],
    "fact_check_urls": ["https://..."]
  },
  "risk_score": 92,
  "risk_level": "CRITICAL",
  "counter_narrative": {
    "summary": "This claim is false...",
    "inconsistencies": ["..."],
    "verified_alternative": "...",
    "advice": "...",
    "citations": ["WHO", "CDC"]
  },
  "detection_signals": {
    "ai_generated_prob": 0.31,
    "deepfake_prob": 0.0,
    "source_credibility": 0.5,
    "nli_false_prob": 0.89,
    "nli_misleading_prob": 0.07
  },
  "explainability": {
    "top_features": [
      {"feature": "fact_check_result", "contribution": 37.6},
      {"feature": "nli_analysis", "contribution": 8.9}
    ],
    "risk_breakdown": {
      "total_score": 92,
      "level": "CRITICAL",
      "components": {}
    }
  },
  "cached": false,
  "processing_time_ms": 1240.5
}
```

---

### `POST /api/v1/analyze/url`
Fetch and analyse a news article URL.

```json
{ "url": "https://example.com/news-article", "target_language": "en" }
```

---

### `POST /api/v1/analyze/batch`
Analyse up to 5 texts at once.

```json
{ "texts": ["claim 1", "claim 2"], "target_language": "en" }
```

---

## Fact Check

### `POST /api/v1/fact-check`
Direct fact-check (no counter-narrative generation).

```json
{ "claim": "The earth is flat", "language": "en" }
```

### `GET /api/v1/fact-check/sources`
List all fact-check sources used in the pipeline.

---

## Counter Narrative

### `POST /api/v1/counter`
Generate counter-narrative for a known claim + verdict.

```json
{
  "claim": "Vaccines cause autism",
  "verdict": "FAKE",
  "sources": ["WHO", "CDC"],
  "target_language": "hi"
}
```

---

## Multimodal

### `POST /api/v1/analyze/image`
Multipart upload. Fields: `file` (image), `caption` (text), `target_language`.

### `POST /api/v1/analyze/audio`
Multipart upload. Fields: `file` (audio), `target_language`.  
Whisper transcribes, then full pipeline runs on transcript.

### `POST /api/v1/analyze/video`
Multipart upload. Fields: `file` (video), `caption`, `target_language`.

### `POST /api/v1/detect/deepfake`
Quick deepfake check — image only. Returns `deepfake_probability`.

### `POST /api/v1/detect/ai-text`
Quick AI text check. Body: `{"text": "..."}`.

---

## Trending & Unique Features

### `GET /api/v1/trending?limit=10`
🔥 Returns most-looked-up misinformation claims, ranked by hit count.

### `POST /api/v1/predict/virality`
🚀 Predict how viral a misinformation claim might become.

```json
{ "claim": "...", "verdict": "FAKE", "risk_score": 90 }
```

### `POST /api/v1/network`
🕸️ Find semantically similar claims (narrative cluster analysis).

```json
{ "claim": "...", "top_k": 5 }
```

### `GET /api/v1/stats`
📊 System stats: total indexed, verdict distribution, top categories.

---

## Segmentation (Special Track)

### `GET /api/v1/segmentation/classes`
List all 10 Duality AI segmentation classes.

### `POST /api/v1/segmentation/analyze`
Upload prediction + ground truth masks → returns per-class IoU.

### `GET /api/v1/segmentation/similar-classes?class_name=Logs`
Find semantically similar classes (for misclassification analysis via Qdrant).

### `GET /api/v1/segmentation/report-template`
Get structured report template for Duality AI submission.

---

## Risk Levels

| Score | Level    | Meaning                              |
|-------|----------|--------------------------------------|
| 0-30  | LOW      | Likely true or low-harm content      |
| 31-60 | MEDIUM   | Misleading — needs context           |
| 61-80 | HIGH     | Likely false — do not share          |
| 81-100| CRITICAL | Confirmed misinformation — dangerous |

## Verdict Types

| Verdict     | Description                              |
|-------------|------------------------------------------|
| TRUE        | Verified as accurate                     |
| FAKE        | Verified as false                        |
| MISLEADING  | Partially true but missing context       |
| UNVERIFIED  | Insufficient data to verify             |
| SATIRE      | Satirical content shared as fact         |
