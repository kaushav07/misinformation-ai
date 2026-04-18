"""
Multilingual support — detect language, translate to English for pipeline,
then optionally translate responses back into the user's language.

Supported natively: Hindi (hi), Tamil (ta), English (en)
Others: auto-detected via langdetect, translated via Google Translate HTTP fallback.
"""
import re
import asyncio
from typing import Tuple
from loguru import logger
from app.models.enums import Language


# ─── Language detection ───────────────────────────────────────────────────────

def detect_language(text: str) -> Language:
    """
    Fast rule-based detection for Hindi/Tamil,
    falls back to langdetect for everything else.
    """
    # Devanagari Unicode block → Hindi
    if re.search(r'[\u0900-\u097F]', text):
        return Language.HINDI

    # Tamil Unicode block
    if re.search(r'[\u0B80-\u0BFF]', text):
        return Language.TAMIL

    try:
        from langdetect import detect
        code = detect(text)
        if code == 'hi':
            return Language.HINDI
        if code == 'ta':
            return Language.TAMIL
        if code == 'en':
            return Language.ENGLISH
    except Exception:
        pass

    return Language.ENGLISH


def is_hindi(text: str) -> bool:
    return bool(re.search(r'[\u0900-\u097F]', text))


def is_tamil(text: str) -> bool:
    return bool(re.search(r'[\u0B80-\u0BFF]', text))


# ─── Translation engine ───────────────────────────────────────────────────────

def _google_translate_free(text: str, src: str, dest: str = "en") -> str:
    """
    Uses the undocumented Google Translate endpoint (no API key needed).
    Reliable enough for hackathon use.
    """
    try:
        import httpx
        import json
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": src,
            "tl": dest,
            "dt": "t",
            "q": text,
        }
        resp = httpx.get(url, params=params, timeout=8)
        data = resp.json()
        translated = "".join(part[0] for part in data[0] if part[0])
        return translated.strip()
    except Exception as e:
        logger.warning(f"Google Translate failed: {e}")
        return text


def translate_to_english(text: str, source_lang: Language = None) -> Tuple[str, Language]:
    """
    Returns (translated_text, detected_language).
    If already English, returns (text, Language.ENGLISH) unchanged.
    """
    detected = source_lang or detect_language(text)

    if detected == Language.ENGLISH:
        return text, Language.ENGLISH

    lang_code = "hi" if detected == Language.HINDI else "ta"
    translated = _google_translate_free(text, src=lang_code, dest="en")
    logger.info(f"Translated from {detected.value}: '{text[:50]}' → '{translated[:50]}'")
    return translated, detected


def translate_response(text: str, target_lang: Language) -> str:
    """
    Translate API response text back to the user's language.
    """
    if target_lang == Language.ENGLISH or target_lang == Language.UNKNOWN:
        return text

    lang_code = "hi" if target_lang == Language.HINDI else "ta"
    return _google_translate_free(text, src="en", dest=lang_code)
