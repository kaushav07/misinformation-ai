"""
All Pydantic request / response models used across the API.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from app.models.enums import Verdict, Language, ContentType, RiskLevel


# ─── Requests ────────────────────────────────────────────────────────────────

class TextAnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Text / claim to analyse")
    target_language: Language = Language.ENGLISH
    include_counter: bool = True

class URLAnalyzeRequest(BaseModel):
    url: str = Field(..., description="URL of the article / post to analyse")
    target_language: Language = Language.ENGLISH

class FactCheckRequest(BaseModel):
    claim: str = Field(..., min_length=5)
    language: Language = Language.ENGLISH

class CounterRequest(BaseModel):
    claim: str
    verdict: Verdict
    sources: List[str] = []
    target_language: Language = Language.ENGLISH

class BatchAnalyzeRequest(BaseModel):
    texts: List[str] = Field(..., max_length=5)
    target_language: Language = Language.ENGLISH


# ─── Sub-models ──────────────────────────────────────────────────────────────

class ClaimInfo(BaseModel):
    claim: str
    claim_type: str = "factual"
    entities: List[str] = []

class FactCheckResult(BaseModel):
    verdict: Verdict
    confidence: float
    explanation: str
    sources: List[str] = []
    fact_check_urls: List[str] = []

class CounterNarrative(BaseModel):
    summary: str
    inconsistencies: List[str] = []
    verified_alternative: str = ""
    advice: str
    citations: List[str] = []

class DetectionSignals(BaseModel):
    ai_generated_prob: float = 0.0
    deepfake_prob: float = 0.0
    source_credibility: float = 0.5
    nli_false_prob: float = 0.0
    nli_misleading_prob: float = 0.0

class ExplainabilityInfo(BaseModel):
    top_features: List[dict] = []
    risk_breakdown: dict = {}

class SegmentationResult(BaseModel):
    iou_score: Optional[float] = None
    class_scores: dict = {}
    failure_cases: List[str] = []


# ─── Main response ───────────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    # Input info
    original_text: str
    detected_language: Language
    translated_text: Optional[str] = None
    content_type: ContentType = ContentType.TEXT

    # Core results
    claim: ClaimInfo
    fact_check: FactCheckResult
    risk_score: int = Field(..., ge=0, le=100)
    risk_level: RiskLevel

    # Counter-narrative (the twist)
    counter_narrative: CounterNarrative

    # ML signals
    detection_signals: DetectionSignals

    # Explainability
    explainability: ExplainabilityInfo

    # Meta
    cached: bool = False
    processing_time_ms: float = 0.0

class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict

class BatchAnalysisResponse(BaseModel):
    results: List[AnalysisResponse]
    total: int
    processing_time_ms: float

class TrendingMisinfoItem(BaseModel):
    claim: str
    verdict: Verdict
    risk_score: int
    hit_count: int
    first_seen: str
    category: str

class TrendingResponse(BaseModel):
    trending: List[TrendingMisinfoItem]
    total_indexed: int

class ViralityPrediction(BaseModel):
    claim: str
    virality_score: float
    predicted_reach: str
    spread_velocity: str
    risk_amplification: float

class NetworkAnalysis(BaseModel):
    claim: str
    similar_claims: List[dict]
    cluster_id: Optional[str]
    narrative_family: str
