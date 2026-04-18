"""
Risk scorer with LIME-style explainability.
Fixed verdict → risk mapping to be intuitive and accurate.
"""
from app.models.schemas import DetectionSignals, ExplainabilityInfo
from app.models.enums import Verdict, RiskLevel

WEIGHTS = {
    "fact_check": 0.40,
    "ai_generated": 0.20,
    "deepfake": 0.15,
    "source_credibility": 0.15,
    "nli": 0.10,
}

# Verdict → base risk (0.0 to 1.0), scaled by confidence
VERDICT_RISK = {
    Verdict.FAKE:        1.00,
    Verdict.MISLEADING:  0.65,
    Verdict.SATIRE:      0.25,
    Verdict.UNVERIFIED:  0.35,
    Verdict.TRUE:        0.05,
}


def compute_risk_score(
    verdict: Verdict,
    confidence: float,
    signals: DetectionSignals,
) -> tuple:
    """
    Returns (risk_score: int, risk_level: RiskLevel, explainability: ExplainabilityInfo).
    """
    # Clamp confidence to valid range
    confidence = max(0.0, min(1.0, float(confidence)))

    # ── Component contributions ───────────────────────────────────
    fact_contrib    = VERDICT_RISK.get(verdict, 0.35) * confidence * WEIGHTS["fact_check"]
    ai_contrib      = float(signals.ai_generated_prob or 0) * WEIGHTS["ai_generated"]
    df_contrib      = float(signals.deepfake_prob or 0) * WEIGHTS["deepfake"]
    cred_contrib    = (1.0 - float(signals.source_credibility or 0.5)) * WEIGHTS["source_credibility"]
    nli_contrib     = (
        float(signals.nli_false_prob or 0) * 0.7 +
        float(signals.nli_misleading_prob or 0) * 0.3
    ) * WEIGHTS["nli"]

    raw = fact_contrib + ai_contrib + df_contrib + cred_contrib + nli_contrib
    risk_score = max(0, min(100, int(round(raw * 100))))

    # ── Risk level ────────────────────────────────────────────────
    if risk_score <= 30:
        level = RiskLevel.LOW
    elif risk_score <= 60:
        level = RiskLevel.MEDIUM
    elif risk_score <= 80:
        level = RiskLevel.HIGH
    else:
        level = RiskLevel.CRITICAL

    # ── Explainability (LIME-style feature attribution) ───────────
    components = {
        "fact_check_result":       round(fact_contrib * 100, 1),
        "ai_generated_content":    round(ai_contrib * 100, 1),
        "deepfake_signal":         round(df_contrib * 100, 1),
        "source_credibility_gap":  round(cred_contrib * 100, 1),
        "nli_analysis":            round(nli_contrib * 100, 1),
    }

    top_features = sorted(
        [{"feature": k, "contribution": v} for k, v in components.items()],
        key=lambda x: x["contribution"],
        reverse=True,
    )

    explainability = ExplainabilityInfo(
        top_features=top_features,
        risk_breakdown={
            "total_score": risk_score,
            "level": level.value,
            "components": components,
        },
    )

    return risk_score, level, explainability