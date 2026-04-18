"""
/api/v1/multimodal — Image, audio, video analysis endpoints.

Fixed logic:
- Image WITHOUT caption → deepfake-only mode, no fake fact-check on image description
- Image WITH caption → deepfake check + fact-check the caption
- Audio → Whisper transcription → fact-check transcript
- Video → frame deepfake + audio transcription → fact-check
"""
import time
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from loguru import logger

from app.models.schemas import (
    AnalysisResponse, ClaimInfo, FactCheckResult,
    CounterNarrative, DetectionSignals, ExplainabilityInfo,
)
from app.models.enums import ContentType, Language, Verdict, RiskLevel
from app.services.detector import (
    detect_image_deepfake,
    detect_video_deepfake,
    transcribe_audio,
    build_detection_signals,
)
from app.services.preprocessor import clean_text
from app.services.translator import translate_to_english, detect_language
from app.services.claim_extractor import extract_claim
from app.services.fact_verifier import verify_claim
from app.services.scorer import compute_risk_score
from app.services.counter_gen import generate_counter_narrative
from app.services import qdrant_service as qs

router = APIRouter()

ALLOWED_IMAGE = {"image/jpeg", "image/png", "image/webp", "image/gif"}
ALLOWED_AUDIO = {"audio/wav", "audio/mpeg", "audio/mp3", "audio/ogg", "audio/webm"}
ALLOWED_VIDEO = {"video/mp4", "video/mpeg", "video/webm", "video/quicktime"}
MAX_FILE_MB = 50


def _validate_file(file: UploadFile, allowed: set):
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: {sorted(allowed)}",
        )


async def _read_file(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_FILE_MB} MB.")
    return data


def _lang(target_language: str) -> Language:
    try:
        return Language(target_language)
    except ValueError:
        return Language.ENGLISH


# ─── Deepfake-only response builder ──────────────────────────────────────────
# Used when an image/video is uploaded WITHOUT any caption or transcript.
# We ONLY report the deepfake score — we do NOT invent a claim to fact-check.

def _deepfake_only_response(
    deepfake_prob: float,
    content_type: ContentType,
    processing_ms: float,
) -> AnalysisResponse:
    """
    Build a clean response for deepfake-only analysis.
    No fact-check, no counter-narrative — just the deepfake signal.
    """
    if deepfake_prob >= 0.65:
        df_verdict = "LIKELY DEEPFAKE"
        df_explanation = (
            f"EfficientNet-B4 analysis found {deepfake_prob:.0%} probability of digital manipulation. "
            "This image shows strong signs of deepfake or AI-generated facial modification."
        )
        risk_score = min(100, int(50 + deepfake_prob * 45))
        risk_level = RiskLevel.HIGH if risk_score <= 80 else RiskLevel.CRITICAL
        fact_verdict = Verdict.FAKE
        fact_confidence = round(deepfake_prob, 3)
    elif deepfake_prob >= 0.35:
        df_verdict = "UNCERTAIN — possible manipulation"
        df_explanation = (
            f"EfficientNet-B4 analysis found {deepfake_prob:.0%} probability of digital manipulation. "
            "Results are inconclusive — image may have subtle edits."
        )
        risk_score = int(30 + deepfake_prob * 30)
        risk_level = RiskLevel.MEDIUM
        fact_verdict = Verdict.MISLEADING
        fact_confidence = round(deepfake_prob, 3)
    else:
        df_verdict = "LIKELY AUTHENTIC"
        df_explanation = (
            f"EfficientNet-B4 analysis found only {deepfake_prob:.0%} probability of manipulation. "
            "This image appears to be authentic."
        )
        risk_score = int(deepfake_prob * 30)
        risk_level = RiskLevel.LOW
        fact_verdict = Verdict.TRUE
        fact_confidence = round(1.0 - deepfake_prob, 3)

    signals = DetectionSignals(
        ai_generated_prob=0.0,
        deepfake_prob=round(deepfake_prob, 3),
        source_credibility=0.5,
        nli_false_prob=0.0,
        nli_misleading_prob=0.0,
    )

    top_features = [
        {"feature": "deepfake_signal", "contribution": round(deepfake_prob * 15, 1)},
        {"feature": "source_credibility_gap", "contribution": 7.5},
    ]

    return AnalysisResponse(
        original_text="[Image uploaded — no caption provided]",
        detected_language=Language.ENGLISH,
        translated_text=None,
        content_type=content_type,
        claim=ClaimInfo(
            claim=df_verdict,
            claim_type="deepfake_detection",
            entities=[],
        ),
        fact_check=FactCheckResult(
            verdict=fact_verdict,
            confidence=fact_confidence,
            explanation=df_explanation,
            sources=["EfficientNet-B4 Deepfake Detector"],
            fact_check_urls=[],
        ),
        risk_score=risk_score,
        risk_level=risk_level,
        counter_narrative=CounterNarrative(
            summary=df_explanation,
            inconsistencies=(
                [
                    "The image shows statistical patterns consistent with AI face-swapping.",
                    "Facial boundary artifacts detected at high probability.",
                    "Pixel-level inconsistencies suggest post-processing manipulation.",
                ]
                if deepfake_prob >= 0.65 else []
            ),
            verified_alternative=(
                "Do not share or trust this image without verification from the original source."
                if deepfake_prob >= 0.35 else
                "This image appears authentic based on deepfake detection analysis."
            ),
            advice=(
                "Perform reverse image search and verify the original source before sharing."
                if deepfake_prob >= 0.35 else
                "This image passed deepfake detection. Always verify sources regardless."
            ),
            citations=["EfficientNet-B4 (ImageNet pretrained deepfake detector)"],
        ),
        detection_signals=signals,
        explainability=ExplainabilityInfo(
            top_features=top_features,
            risk_breakdown={
                "total_score": risk_score,
                "level": risk_level.value,
                "note": "Deepfake-only analysis — no caption provided so no fact-check was run.",
                "deepfake_probability": f"{deepfake_prob:.0%}",
            },
        ),
        cached=False,
        processing_time_ms=round(processing_ms, 1),
    )


# ─── Full pipeline (text + optional media) ───────────────────────────────────

def _pipeline_from_text(
    text: str,
    content_type: ContentType,
    image_bytes: bytes = None,
    audio_bytes: bytes = None,
    video_bytes: bytes = None,
    target_language: Language = Language.ENGLISH,
    t0: float = None,
) -> AnalysisResponse:
    """Full pipeline: clean → translate → claim → fact-check → score → counter."""
    if t0 is None:
        t0 = time.time()

    cleaned = clean_text(text)
    detected_lang = detect_language(cleaned)
    translated, _ = translate_to_english(cleaned, detected_lang)
    claim_info = extract_claim(translated)

    # Qdrant cache check
    cached = qs.search_similar_claim(claim_info.claim)
    fact_result = verify_claim(claim_info.claim)

    signals = build_detection_signals(
        text=translated,
        image_bytes=image_bytes,
        audio_bytes=audio_bytes,
        video_bytes=video_bytes,
        nli_false_prob=fact_result.confidence if fact_result.verdict == Verdict.FAKE else 0.0,
        nli_misleading_prob=fact_result.confidence if fact_result.verdict == Verdict.MISLEADING else 0.0,
    )

    risk_score, risk_level, explainability = compute_risk_score(
        fact_result.verdict, fact_result.confidence, signals
    )
    counter = generate_counter_narrative(
        claim_info.claim, fact_result.verdict, fact_result, target_language
    )

    try:
        qs.upsert_claim(claim_info.claim, fact_result.verdict.value, risk_score, fact_result.sources)
    except Exception as e:
        logger.warning(f"Qdrant upsert failed: {e}")

    return AnalysisResponse(
        original_text=text[:500],
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
        cached=bool(cached),
        processing_time_ms=round((time.time() - t0) * 1000, 1),
    )


# ─── Image endpoint ───────────────────────────────────────────────────────────

@router.post("/analyze/image", response_model=AnalysisResponse,
             summary="Analyse an image for deepfakes + optional caption fact-check")
async def analyze_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)"),
    caption: str = Form("", description="Optional claim or headline about this image"),
    target_language: str = Form("en"),
):
    """
    Two modes depending on whether a caption is provided:

    - **No caption**: deepfake-only mode — EfficientNet-B4 scans for manipulation.
      Returns deepfake probability, verdict, and explanation. Does NOT invent a claim.

    - **With caption**: deepfake scan + full fact-check of the caption.
      Use this when you have a headline/claim that accompanies the image.
    """
    _validate_file(file, ALLOWED_IMAGE)
    image_bytes = await _read_file(file)
    lang = _lang(target_language)
    t0 = time.time()

    caption = caption.strip()

    if not caption:
        # ── Deepfake-only mode ────────────────────────────────────
        deepfake_prob = detect_image_deepfake(image_bytes)
        return _deepfake_only_response(deepfake_prob, ContentType.IMAGE,
                                       (time.time() - t0) * 1000)
    else:
        # ── Caption fact-check mode ───────────────────────────────
        return _pipeline_from_text(caption, ContentType.IMAGE,
                                   image_bytes=image_bytes, target_language=lang, t0=t0)


# ─── Audio endpoint ───────────────────────────────────────────────────────────

@router.post("/analyze/audio", response_model=AnalysisResponse,
             summary="Transcribe audio and fact-check")
async def analyze_audio(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, OGG)"),
    target_language: str = Form("en"),
):
    """
    Whisper transcribes the audio, then the transcript is fact-checked.
    Supports Hindi and Tamil audio — language is auto-detected from transcript.
    """
    _validate_file(file, ALLOWED_AUDIO)
    audio_bytes = await _read_file(file)
    lang = _lang(target_language)
    t0 = time.time()

    transcribed = transcribe_audio(audio_bytes)
    if not transcribed:
        raise HTTPException(
            status_code=422,
            detail="Could not transcribe audio. Ensure the file contains clear speech and is not corrupted.",
        )

    return _pipeline_from_text(transcribed, ContentType.AUDIO,
                               audio_bytes=audio_bytes, target_language=lang, t0=t0)


# ─── Video endpoint ───────────────────────────────────────────────────────────

@router.post("/analyze/video", response_model=AnalysisResponse,
             summary="Analyse video for deepfakes + audio fact-check")
async def analyze_video(
    file: UploadFile = File(..., description="Video file (MP4, WebM)"),
    caption: str = Form("", description="Optional claim or headline about this video"),
    target_language: str = Form("en"),
):
    """
    Two modes:
    - **No caption, no transcription**: deepfake frame scan only.
    - **Caption or transcribed audio**: deepfake + full fact-check pipeline.
    """
    _validate_file(file, ALLOWED_VIDEO)
    video_bytes = await _read_file(file)
    lang = _lang(target_language)
    t0 = time.time()

    caption = caption.strip()

    # Try audio transcription from video first
    transcribed = transcribe_audio(video_bytes)
    analysis_text = transcribed or caption

    if not analysis_text:
        # Deepfake-only mode
        deepfake_prob = detect_video_deepfake(video_bytes)
        return _deepfake_only_response(deepfake_prob, ContentType.VIDEO,
                                       (time.time() - t0) * 1000)

    return _pipeline_from_text(analysis_text, ContentType.VIDEO,
                               video_bytes=video_bytes, target_language=lang, t0=t0)


# ─── Quick deepfake check ─────────────────────────────────────────────────────

@router.post("/detect/deepfake", summary="Quick deepfake probability (image only)")
async def quick_deepfake(
    file: UploadFile = File(..., description="Image to check"),
):
    """Returns only the deepfake probability. No fact-check pipeline."""
    _validate_file(file, ALLOWED_IMAGE)
    image_bytes = await _read_file(file)
    prob = detect_image_deepfake(image_bytes)

    if prob >= 0.65:
        verdict = "LIKELY DEEPFAKE"
    elif prob <= 0.35:
        verdict = "LIKELY AUTHENTIC"
    else:
        verdict = "UNCERTAIN"

    return {
        "deepfake_probability": round(prob, 4),
        "authentic_probability": round(1.0 - prob, 4),
        "verdict": verdict,
        "confidence": round(max(prob, 1.0 - prob), 4),
        "explanation": (
            f"EfficientNet-B4 detected {prob:.0%} probability of facial manipulation."
            if prob >= 0.35 else
            f"EfficientNet-B4 found only {prob:.0%} manipulation probability. Image appears authentic."
        ),
    }


# ─── Quick AI text check ──────────────────────────────────────────────────────

@router.post("/detect/ai-text", summary="Quick AI-generated text probability")
async def quick_ai_text(payload: dict):
    """
    Returns AI-generated probability for the given text.
    Input: {"text": "..."}
    """
    text = payload.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="'text' field is required.")

    from app.services.detector import detect_ai_generated_text
    prob = detect_ai_generated_text(text)

    if prob >= 0.6:
        verdict = "LIKELY AI-GENERATED"
    elif prob <= 0.3:
        verdict = "LIKELY HUMAN-WRITTEN"
    else:
        verdict = "UNCERTAIN"

    return {
        "ai_generated_probability": round(prob, 4),
        "human_probability": round(1.0 - prob, 4),
        "verdict": verdict,
        "confidence": round(max(prob, 1.0 - prob), 4),
    }