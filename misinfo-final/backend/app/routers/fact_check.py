"""
/api/v1/fact-check — Direct fact-checking endpoints.
Exposes the fact verification pipeline independently of the full analyze flow.
"""
from fastapi import APIRouter, HTTPException
from app.models.schemas import FactCheckRequest, FactCheckResult
from app.models.enums import Language
from app.services.translator import translate_to_english, detect_language
from app.services.preprocessor import clean_text
from app.services.fact_verifier import verify_claim

router = APIRouter()


@router.post("/fact-check", response_model=FactCheckResult, summary="Fact-check a single claim")
def fact_check_claim(req: FactCheckRequest):
    """
    Directly fact-check a claim using the multi-source verification pipeline:
    Google Fact Check API → Wikipedia + NLI → Serper + NLI → NLI fallback.
    """
    cleaned = clean_text(req.claim)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Claim is empty after cleaning.")

    # Translate if needed
    lang = detect_language(cleaned)
    if lang != Language.ENGLISH:
        translated, _ = translate_to_english(cleaned, lang)
    else:
        translated = cleaned

    return verify_claim(translated)


@router.get("/fact-check/sources", summary="List supported fact-check sources")
def list_sources():
    """Returns the list of sources used in the fact-check pipeline."""
    return {
        "sources": [
            {
                "name": "Google Fact Check Tools API",
                "type": "authoritative",
                "priority": 1,
                "description": "Official fact-checks from publishers like PolitiFact, Snopes, AFP, BOOM, etc.",
            },
            {
                "name": "Wikipedia",
                "type": "encyclopaedic",
                "priority": 2,
                "description": "Contextual information retrieved and cross-checked via NLI entailment.",
            },
            {
                "name": "Serper Web Search",
                "type": "web",
                "priority": 3,
                "description": "Real-time Google search snippets, cross-checked via NLI.",
            },
            {
                "name": "BART-large-MNLI (NLI Model)",
                "type": "ml_model",
                "priority": 4,
                "description": "Zero-shot natural language inference fallback classifier.",
            },
            {
                "name": "Qdrant Semantic Cache",
                "type": "vector_db",
                "priority": 0,
                "description": "Previously verified claims retrieved by cosine similarity (threshold 0.85).",
            },
        ]
    }
