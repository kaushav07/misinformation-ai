"""
Claim extractor — pulls the core factual claim from raw text.

Pipeline:
1. Sentence split
2. Rank sentences by "claim-ness" (presence of named entities, verbs, numbers)
3. Return the top candidate as the canonical claim

Also extracts named entities for explainability.
"""
import re
from typing import List, Tuple
from loguru import logger
from app.models.schemas import ClaimInfo


# ─── Sentence splitter ───────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


# ─── Claim score heuristic ───────────────────────────────────────────────────

_CLAIM_KEYWORDS = [
    'causes', 'proven', 'study', 'research', 'scientists', 'government',
    'vaccine', 'cancer', 'cure', 'kills', 'spread', 'billion', 'million',
    'percent', '%', 'confirm', 'reveal', 'warn', 'show', 'find', 'report',
    'never', 'always', 'all', 'none', 'every', 'fake', 'true', 'false',
    'breaking', 'official', 'secret', 'hidden', 'expose', 'ban', 'law',
    'election', 'vote', 'president', 'prime minister', 'minister',
]

_ENTITY_PATTERN = re.compile(
    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|'   # Proper nouns
    r'\b\d{4}\b|'                               # Years
    r'\b\d+(?:\.\d+)?%\b|'                     # Percentages
    r'\b(?:COVID|WHO|UN|NASA|FBI|CIA|BJP|INC|ISRO)\b'  # Acronyms
)


def _score_sentence(sentence: str) -> float:
    score = 0.0
    lower = sentence.lower()

    # Keyword matches
    for kw in _CLAIM_KEYWORDS:
        if kw in lower:
            score += 1.0

    # Numbers add credibility-sounding weight
    numbers = re.findall(r'\d+', sentence)
    score += len(numbers) * 0.5

    # Named entities
    entities = _ENTITY_PATTERN.findall(sentence)
    score += len(entities) * 0.8

    # Prefer medium-length sentences (not too short, not too long)
    words = len(sentence.split())
    if 8 <= words <= 40:
        score += 2.0

    return score


def _extract_entities(text: str) -> List[str]:
    matches = _ENTITY_PATTERN.findall(text)
    # Deduplicate preserving order
    seen = set()
    result = []
    for m in matches:
        if m not in seen and len(m) > 1:
            seen.add(m)
            result.append(m)
    return result[:10]


def _classify_claim_type(text: str) -> str:
    lower = text.lower()
    if any(w in lower for w in ['vaccine', 'disease', 'cancer', 'cure', 'health', 'drug']):
        return "health"
    if any(w in lower for w in ['election', 'vote', 'president', 'minister', 'government', 'party']):
        return "political"
    if any(w in lower for w in ['climate', 'global warming', 'environment', 'pollution']):
        return "environmental"
    if any(w in lower for w in ['economy', 'gdp', 'inflation', 'market', 'bank', 'rupee']):
        return "economic"
    if any(w in lower for w in ['religion', 'temple', 'mosque', 'church', 'god', 'prophet']):
        return "religious"
    return "general"


# ─── Public API ──────────────────────────────────────────────────────────────

def extract_claim(text: str) -> ClaimInfo:
    """
    Extract the core falsifiable claim from text.
    Returns a ClaimInfo with claim, type, and named entities.
    """
    if not text or len(text.strip()) < 10:
        return ClaimInfo(claim=text, claim_type="general", entities=[])

    sentences = _split_sentences(text)

    if not sentences:
        return ClaimInfo(
            claim=text[:300],
            claim_type=_classify_claim_type(text),
            entities=_extract_entities(text)
        )

    # Score each sentence
    scored = [(s, _score_sentence(s)) for s in sentences]
    scored.sort(key=lambda x: x[1], reverse=True)

    best_claim = scored[0][0]
    entities = _extract_entities(text)
    claim_type = _classify_claim_type(text)

    logger.debug(f"Extracted claim: '{best_claim[:80]}' | type={claim_type}")

    return ClaimInfo(
        claim=best_claim,
        claim_type=claim_type,
        entities=entities
    )
