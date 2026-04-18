"""
/api/v1/counter — Counter-narrative generation endpoints.
"""
from fastapi import APIRouter, HTTPException
from app.models.schemas import CounterRequest, CounterNarrative, FactCheckResult
from app.models.enums import Verdict
from app.services.counter_gen import generate_counter_narrative

router = APIRouter()


@router.post("/counter", response_model=CounterNarrative, summary="Generate counter-narrative for a claim")
def generate_counter(req: CounterRequest):
    """
    Generate a structured counter-narrative for a given claim and verdict.

    - Uses Claude API if configured, else rule-based templates
    - Supports response in Hindi, Tamil, or English
    - Returns: summary, inconsistencies, verified alternative, advice, citations
    """
    if not req.claim.strip():
        raise HTTPException(status_code=400, detail="Claim cannot be empty.")

    # Build a minimal FactCheckResult to pass context
    fact_result = FactCheckResult(
        verdict=req.verdict,
        confidence=0.85,
        explanation="Provided via direct counter-narrative request.",
        sources=req.sources,
    )

    return generate_counter_narrative(
        claim=req.claim,
        verdict=req.verdict,
        fact_result=fact_result,
        target_language=req.target_language,
    )


@router.get("/counter/templates", summary="Get available counter-narrative templates")
def list_templates():
    """Returns the verdict types and what kind of counter-narrative each produces."""
    return {
        "templates": [
            {
                "verdict": "FAKE",
                "summary": "Flags as false, lists specific contradictions, provides verified alternative.",
                "languages": ["en", "hi", "ta"],
            },
            {
                "verdict": "MISLEADING",
                "summary": "Acknowledges partial truth, highlights missing context, provides full picture.",
                "languages": ["en", "hi", "ta"],
            },
            {
                "verdict": "TRUE",
                "summary": "Confirms accuracy, reinforces with credible citations.",
                "languages": ["en", "hi", "ta"],
            },
            {
                "verdict": "UNVERIFIED",
                "summary": "Advises caution, explains why verification failed.",
                "languages": ["en", "hi", "ta"],
            },
            {
                "verdict": "SATIRE",
                "summary": "Identifies satirical nature, warns against sharing as fact.",
                "languages": ["en"],
            },
        ]
    }
