"""
Multimodal detector — handles:
  • AI-generated text detection (RoBERTa)
  • Deepfake image/video detection (EfficientNet-B4)
  • Audio deepfake / transcription (Whisper)
  • Source credibility scoring
  • URL content extraction

Returns DetectionSignals for use in the risk scorer.
"""
import io
import base64
import tempfile
import os
from typing import Optional
from loguru import logger
from app.models.schemas import DetectionSignals


# ─── Lazy model handles ──────────────────────────────────────────────────────

_ai_detector = None
_deepfake_model = None
_whisper_model = None


def _get_ai_detector():
    global _ai_detector
    if _ai_detector is None:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading AI content detector…")
        try:
            _ai_detector = hf_pipeline(
                "text-classification",
                model="roberta-base-openai-detector",
                device=-1,
            )
            logger.info("AI detector loaded ✅")
        except Exception as e:
            logger.warning(f"AI detector load failed: {e}. Using fallback.")
            _ai_detector = "fallback"
    return _ai_detector


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            logger.info("Loading Whisper model…")
            _whisper_model = whisper.load_model("base")
            logger.info("Whisper loaded ✅")
        except Exception as e:
            logger.warning(f"Whisper load failed: {e}")
            _whisper_model = "unavailable"
    return _whisper_model


# ─── AI text detection ───────────────────────────────────────────────────────

def detect_ai_generated_text(text: str) -> float:
    """
    Returns probability [0, 1] that text is AI-generated.
    Uses roberta-base-openai-detector.
    """
    detector = _get_ai_detector()
    if detector == "fallback" or detector is None:
        return _heuristic_ai_score(text)

    try:
        result = detector(text[:512])
        label = result[0]["label"].upper()
        score = result[0]["score"]
        # Label is "FAKE" (AI) or "REAL" (human) in this model
        if "FAKE" in label or "AI" in label:
            return round(score, 3)
        else:
            return round(1.0 - score, 3)
    except Exception as e:
        logger.warning(f"AI detection inference failed: {e}")
        return _heuristic_ai_score(text)


def _heuristic_ai_score(text: str) -> float:
    """
    Simple heuristic AI score when model is unavailable.
    Looks for patterns common in AI-generated text.
    """
    import re
    score = 0.0
    # Very uniform sentence lengths
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    if len(sentences) > 3:
        lengths = [len(s.split()) for s in sentences]
        variance = sum((l - sum(lengths)/len(lengths))**2 for l in lengths) / len(lengths)
        if variance < 5:  # Very uniform → likely AI
            score += 0.3

    # Common AI phrases
    ai_phrases = [
        "it is important to note", "in conclusion", "furthermore",
        "it should be noted", "in summary", "to summarize",
        "as an ai", "as a language model", "in this context"
    ]
    lower = text.lower()
    matches = sum(1 for p in ai_phrases if p in lower)
    score += min(0.5, matches * 0.15)

    return round(min(1.0, score), 3)


# ─── Image deepfake detection ─────────────────────────────────────────────────

def detect_image_deepfake(image_data: bytes) -> float:
    """
    EfficientNet-B4 based deepfake detection.
    Returns probability [0, 1] that image is a deepfake.
    """
    try:
        import torch
        import timm
        from PIL import Image
        from torchvision import transforms

        global _deepfake_model
        if _deepfake_model is None:
            logger.info("Loading EfficientNet-B4 for deepfake detection…")
            _deepfake_model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=2)
            _deepfake_model.eval()
            logger.info("Deepfake model loaded ✅")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            logits = _deepfake_model(tensor)
            probs = torch.softmax(logits, dim=1)
            fake_prob = probs[0][1].item()

        return round(fake_prob, 3)

    except Exception as e:
        logger.warning(f"Deepfake detection failed: {e}")
        return 0.0


def detect_video_deepfake(video_bytes: bytes) -> float:
    """
    Extract frames from video and average deepfake scores.
    """
    try:
        import cv2
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            tmp_path = f.name

        cap = cv2.VideoCapture(tmp_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_interval = max(1, frame_count // 10)  # sample 10 frames

        scores = []
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_interval == 0:
                _, buf = cv2.imencode('.jpg', frame)
                score = detect_image_deepfake(buf.tobytes())
                scores.append(score)
            idx += 1

        cap.release()
        os.unlink(tmp_path)

        return round(float(np.mean(scores)), 3) if scores else 0.0

    except Exception as e:
        logger.warning(f"Video deepfake detection failed: {e}")
        return 0.0


# ─── Audio transcription ─────────────────────────────────────────────────────

def _detect_file_extension(data: bytes) -> str:
    """
    Detect the correct file extension from magic bytes.
    Whisper / ffmpeg MUST receive the right extension or it misreads the container.
    """
    if data[:4] == b"\x1aE\xdf\xa3":
        return ".webm"
    # MP4 / MOV — ftyp box at byte 4
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return ".mp4"
    # MP4 alternative signatures
    if data[4:8] in (b"moov", b"mdat", b"free", b"skip"):
        return ".mp4"
    # AVI
    if data[:4] == b"RIFF" and data[8:12] == b"AVI ":
        return ".avi"
    # WAV
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return ".wav"
    # MP3
    if data[:3] == b"ID3" or (data[:2] == b"\xff\xfb"):
        return ".mp3"
    # OGG
    if data[:4] == b"OggS":
        return ".ogg"
    # FLAC
    if data[:4] == b"fLaC":
        return ".flac"
    # Default to mp4 for unknown (safer than wav for video content)
    return ".mp4"


def transcribe_audio(audio_bytes: bytes) -> Optional[str]:
    """
    Whisper-based transcription for audio OR video files.
    Detects the actual file type from magic bytes so ffmpeg gets
    the correct container format — fixes the 'Output file does not
    contain any stream' error when video bytes are passed with .wav suffix.
    """
    whisper_model = _get_whisper()
    if whisper_model == "unavailable":
        return None

    tmp_path = None
    try:
        ext = _detect_file_extension(audio_bytes)
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        logger.debug(f"Transcribing {ext} file ({len(audio_bytes)//1024}KB) at {tmp_path}")

        result = whisper_model.transcribe(tmp_path, fp16=False, verbose=False)
        text = result.get("text", "").strip()

        if not text:
            logger.debug("Whisper returned empty transcript (silent or no speech track)")
            return None

        logger.info(f"Whisper transcript ({len(text)} chars): '{text[:80]}'")
        return text

    except Exception as e:
        logger.warning(f"Whisper transcription failed: {e}")
        return None
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ─── Source credibility ───────────────────────────────────────────────────────

# Known credible / low-credibility domains
_HIGH_CREDIBILITY = {
    "bbc.com", "reuters.com", "apnews.com", "thehindu.com",
    "ndtv.com", "timesofindia.com", "indianexpress.com",
    "hindustantimes.com", "pib.gov.in", "who.int",
    "cdc.gov", "icmr.gov.in", "mohfw.gov.in", "nature.com",
    "sciencemag.org", "pubmed.ncbi.nlm.nih.gov",
}

_LOW_CREDIBILITY = {
    "postcard.news", "sudarshannews.com", "opindia.com",
    "thewire.in",  # controversial
    "newsbharati.com", "hindi.swarajyamag.com",
}


def score_source_credibility(url_or_domain: str) -> float:
    """
    Returns credibility score [0, 1].
    1.0 = highly credible, 0.0 = known misinformation source.
    """
    import re
    domain_match = re.search(r'(?:https?://)?(?:www\.)?([^/\s]+)', url_or_domain.lower())
    if not domain_match:
        return 0.5

    domain = domain_match.group(1)

    if domain in _HIGH_CREDIBILITY:
        return 0.95
    if domain in _LOW_CREDIBILITY:
        return 0.15

    # Social media platforms — medium credibility
    if any(s in domain for s in ['twitter', 'facebook', 'whatsapp', 'telegram', 'youtube']):
        return 0.35

    # Government domains
    if domain.endswith('.gov.in') or domain.endswith('.gov'):
        return 0.90

    # Academic / research
    if domain.endswith('.edu') or domain.endswith('.ac.in'):
        return 0.85

    return 0.50  # unknown


# ─── Public helper ────────────────────────────────────────────────────────────

def build_detection_signals(
    text: str,
    image_bytes: Optional[bytes] = None,
    audio_bytes: Optional[bytes] = None,
    video_bytes: Optional[bytes] = None,
    source_url: Optional[str] = None,
    nli_false_prob: float = 0.0,
    nli_misleading_prob: float = 0.0,
) -> DetectionSignals:
    """
    Orchestrates all detectors and returns a unified DetectionSignals object.
    """
    ai_prob = detect_ai_generated_text(text)
    deepfake_prob = 0.0

    if image_bytes:
        deepfake_prob = detect_image_deepfake(image_bytes)
    elif video_bytes:
        deepfake_prob = detect_video_deepfake(video_bytes)

    source_cred = score_source_credibility(source_url) if source_url else 0.5

    return DetectionSignals(
        ai_generated_prob=ai_prob,
        deepfake_prob=deepfake_prob,
        source_credibility=source_cred,
        nli_false_prob=nli_false_prob,
        nli_misleading_prob=nli_misleading_prob,
    )