"""
Preprocessor — cleans and normalises any input type before the pipeline.
Handles text, base64 images, audio bytes, and URL content.
"""
import re
import unicodedata
import html
from typing import Optional
from loguru import logger


def clean_text(text: str) -> str:
    """
    Full normalisation:
    - Decode HTML entities
    - Strip URLs
    - Remove special chars but preserve Devanagari, Tamil, Latin
    - Collapse whitespace
    """
    if not text:
        return ""

    # Decode HTML entities (&amp; → &)
    text = html.unescape(text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove mentions and hashtags
    text = re.sub(r'[@#]\w+', '', text)

    # Remove emojis (keep text)
    text = re.sub(
        r'[\U00010000-\U0010ffff]|[\U0001F600-\U0001F64F]|'
        r'[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]',
        '', text, flags=re.UNICODE
    )

    # Normalise unicode (NFC)
    text = unicodedata.normalize('NFC', text)

    # Replace newlines and tabs with space
    text = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def extract_text_from_url(url: str) -> Optional[str]:
    """
    Fetches article text from a URL using httpx + basic HTML stripping.
    Falls back gracefully if fetch fails.
    """
    try:
        import httpx
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text_parts = []
                self._skip = False

            def handle_starttag(self, tag, attrs):
                if tag in ('script', 'style', 'nav', 'footer', 'header'):
                    self._skip = True

            def handle_endtag(self, tag):
                if tag in ('script', 'style', 'nav', 'footer', 'header'):
                    self._skip = False

            def handle_data(self, data):
                if not self._skip:
                    stripped = data.strip()
                    if len(stripped) > 20:
                        self.text_parts.append(stripped)

        resp = httpx.get(url, timeout=10, follow_redirects=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        parser = TextExtractor()
        parser.feed(resp.text)
        raw = ' '.join(parser.text_parts)
        return clean_text(raw)[:3000]  # limit
    except Exception as e:
        logger.warning(f"URL fetch failed for {url}: {e}")
        return None


def truncate_for_model(text: str, max_tokens: int = 512) -> str:
    """
    Rough truncation by character count (avg 4 chars/token).
    """
    max_chars = max_tokens * 4
    if len(text) > max_chars:
        text = text[:max_chars]
        # Don't cut mid-word
        last_space = text.rfind(' ')
        if last_space > max_chars - 50:
            text = text[:last_space]
    return text
