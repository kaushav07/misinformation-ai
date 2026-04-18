"""
/api/v1/analyze — Core analysis endpoint.
Handles text, URL, and batch requests through the full pipeline.
"""
import time
from fastapi import APIRouter, HTTPException
from loguru import logger

from app.models.schemas import (
    TextAnalyzeRequest, URLAnalyzeRequest, BatchAnalyzeRequest,
    AnalysisResponse, BatchAnalysisResponse, ClaimInfo,
)
from app.models.enums import ContentType, Language
from app.services.preprocessor import clean_text, extract_text_from_url
from app.services.translator import translate_to_english, detect_language
from app.services.claim_extractor import extract_claim
from app.services.fact_verifier import verify_claim
from app.services.detector import build_detection_signals
from app.services.scorer import compute_risk_score
from app.services.counter_gen import generate_counter_narrative
from app.services import qdrant_service as qs

router = APIRouter()


def _run_full_pipeline(
    raw_text: str,
    target_language: Language = Language.ENGLISH,
    include_counter: bool = True,
    content_type: ContentType = ContentType.TEXT,
    source_url: str = None,
) -> AnalysisResponse:
    """
    Core pipeline shared across all text-based endpoints:
    clean → detect lang → translate → extract claim →
    qdrant cache check → fact verify → detect signals →
    score → counter-narrative → upsert → return
    """
    t0 = time.time()

    # ── 1. Preprocess ─────────────────────────────────────────────
    cleaned = clean_text(raw_text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Input text is empty after cleaning.")

    # ── 2. Language detection & translation ───────────────────────
    detected_lang = detect_language(cleaned)
    translated, _ = translate_to_english(cleaned, detected_lang)

    # ── 3. Claim extraction ───────────────────────────────────────
    claim_info: ClaimInfo = extract_claim(translated)

    # ── 4. Qdrant cache lookup ────────────────────────────────────
    cached = qs.search_similar_claim(claim_info.claim)
    if cached:
        # Reconstruct response from cache
        from app.models.schemas import FactCheckResult, DetectionSignals, ExplainabilityInfo
        from app.models.enums import Verdict, RiskLevel
        from app.services.scorer import VERDICT_RISK

        cached_verdict = Verdict(cached["verdict"])
        cached_risk = cached.get("risk_score", 50)
        cached_level = (
            RiskLevel.LOW if cached_risk <= 30 else
            RiskLevel.MEDIUM if cached_risk <= 60 else
            RiskLevel.HIGH if cached_risk <= 80 else
            RiskLevel.CRITICAL
        )
        fact_result = FactCheckResult(
            verdict=cached_verdict,
            confidence=0.90,
            explanation="Retrieved from Qdrant semantic cache (previously verified claim).",
            sources=cached.get("sources", ["Qdrant cache"]),
        )
        counter = generate_counter_narrative(
            claim_info.claim, cached_verdict, fact_result, target_language
        ) if include_counter else None

        return AnalysisResponse(
            original_text=raw_text,
            detected_language=detected_lang,
            translated_text=translated if detected_lang != Language.ENGLISH else None,
            content_type=content_type,
            claim=claim_info,
            fact_check=fact_result,
            risk_score=cached_risk,
            risk_level=cached_level,
            counter_narrative=counter,
            detection_signals=DetectionSignals(),
            explainability=ExplainabilityInfo(
                top_features=[],
                risk_breakdown={"note": "Served from Qdrant cache"},
            ),
            cached=True,
            processing_time_ms=round((time.time() - t0) * 1000, 1),
        )

    # ── 5. Fact verification ──────────────────────────────────────
    fact_result = verify_claim(claim_info.claim)

    # ── 6. Detection signals ──────────────────────────────────────
    signals = build_detection_signals(
        text=translated,
        source_url=source_url,
        nli_false_prob=1.0 - fact_result.confidence if fact_result.verdict.value == "FAKE" else 0.0,
        nli_misleading_prob=fact_result.confidence if fact_result.verdict.value == "MISLEADING" else 0.0,
    )

    # ── 7. Risk scoring + explainability ─────────────────────────
    risk_score, risk_level, explainability = compute_risk_score(
        fact_result.verdict, fact_result.confidence, signals
    )

    # ── 8. Counter-narrative ──────────────────────────────────────
    counter = generate_counter_narrative(
        claim_info.claim, fact_result.verdict, fact_result, target_language
    ) if include_counter else None

    # ── 9. Upsert to Qdrant ───────────────────────────────────────
    try:
        qs.upsert_claim(
            claim_info.claim,
            fact_result.verdict.value,
            risk_score,
            fact_result.sources,
        )
    except Exception as e:
        logger.warning(f"Qdrant upsert failed: {e}")

    return AnalysisResponse(
        original_text=raw_text,
        detected_language=detected_lang,
        translated_text=translated if detected_lang != Language.ENGLISH else None,
        content_type=content_type,
        claim=claim_info,
        fact_check=fact_result,
        risk_score=risk_score,
        risk_level=risk_level,
        counter_narrative=counter,
        detection_signals=signals,
        explainability=explainability,
        cached=False,
        processing_time_ms=round((time.time() - t0) * 1000, 1),
    )


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalysisResponse, summary="Analyse a text claim")
def analyze_text(req: TextAnalyzeRequest):
    """
    Full pipeline for text input.
    Supports Hindi, Tamil, and English. Checks Qdrant cache first.
    """
    return _run_full_pipeline(
        raw_text=req.text,
        target_language=req.target_language,
        include_counter=req.include_counter,
    )


@router.post("/analyze/url", response_model=AnalysisResponse, summary="Analyse a news article URL")
def analyze_url(req: URLAnalyzeRequest):
    """
    Fetches article content from a URL, then runs the full pipeline.
    """
    text = extract_text_from_url(req.url)
    if not text:
        raise HTTPException(status_code=422, detail="Could not extract text from the provided URL.")
    return _run_full_pipeline(
        raw_text=text,
        target_language=req.target_language,
        content_type=ContentType.URL,
        source_url=req.url,
    )


@router.post("/analyze/batch", response_model=BatchAnalysisResponse, summary="Analyse up to 5 claims at once")
def analyze_batch(req: BatchAnalyzeRequest):
    """
    Batch analysis — runs each text through the full pipeline independently.
    Maximum 5 texts per request.
    """
    t0 = time.time()
    results = []
    for text in req.texts[:5]:
        try:
            result = _run_full_pipeline(text, req.target_language)
            results.append(result)
        except Exception as e:
            logger.warning(f"Batch item failed: {e}")

    return BatchAnalysisResponse(
        results=results,
        total=len(results),
        processing_time_ms=round((time.time() - t0) * 1000, 1),
    )
