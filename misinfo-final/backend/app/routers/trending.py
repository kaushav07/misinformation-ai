"""
/api/v1/trending — Unique features: trending misinformation, virality prediction,
network analysis, and language-filtered browsing.
"""
from fastapi import APIRouter, Query
from app.models.schemas import TrendingResponse, TrendingMisinfoItem, ViralityPrediction, NetworkAnalysis
from app.models.enums import Verdict
from app.services import qdrant_service as qs

router = APIRouter()


@router.get("/trending", response_model=TrendingResponse, summary="🔥 Trending misinformation claims")
def get_trending(limit: int = Query(10, ge=1, le=50)):
    """
    **UNIQUE FEATURE**: Returns the most-looked-up misinformation claims
    from the Qdrant cache, ranked by hit count.

    Use this to see what misinformation is circulating right now.
    """
    raw = qs.get_trending_claims(limit=limit)

    items = []
    for r in raw:
        try:
            items.append(TrendingMisinfoItem(
                claim=r.get("claim", ""),
                verdict=Verdict(r.get("verdict", "UNVERIFIED")),
                risk_score=r.get("risk_score", 0),
                hit_count=r.get("hit_count", 1),
                first_seen=r.get("first_seen", "unknown"),
                category=r.get("category", "general"),
            ))
        except Exception:
            continue

    # Count total indexed
    try:
        client = qs._get_client()
        total = client.get_collection("claims").points_count
    except Exception:
        total = len(items)

    return TrendingResponse(trending=items, total_indexed=total)


@router.post("/predict/virality", response_model=ViralityPrediction, summary="🚀 Predict misinformation virality")
def predict_virality(payload: dict):
    """
    **UNIQUE FEATURE**: Given a claim + verdict + risk score,
    predicts how viral the misinformation is likely to become based on:
    - Cluster size in Qdrant (how many similar claims already exist)
    - Verdict type amplification factor
    - Risk score baseline

    Returns: virality_score, predicted_reach, spread_velocity
    """
    claim = payload.get("claim", "")
    verdict = payload.get("verdict", "UNVERIFIED")
    risk_score = int(payload.get("risk_score", 50))

    result = qs.predict_virality(claim, verdict, risk_score)

    return ViralityPrediction(
        claim=claim,
        virality_score=result["virality_score"],
        predicted_reach=result["predicted_reach"],
        spread_velocity=result["spread_velocity"],
        risk_amplification=result["risk_amplification"],
    )


@router.post("/network", response_model=NetworkAnalysis, summary="🕸️ Narrative network analysis")
def network_analysis(payload: dict):
    """
    **UNIQUE FEATURE**: Given a claim, finds semantically similar claims
    already indexed, revealing misinformation narrative clusters and families.

    Useful for understanding whether a piece of misinformation is part of
    a coordinated campaign or an isolated incident.
    """
    claim = payload.get("claim", "")
    top_k = int(payload.get("top_k", 5))

    similar = qs.get_similar_claims(claim, top_k=min(top_k, 20))

    # Determine narrative family from most common verdict in cluster
    verdicts = [s.get("verdict", "UNVERIFIED") for s in similar]
    if verdicts:
        from collections import Counter
        most_common_verdict = Counter(verdicts).most_common(1)[0][0]
        families = {
            "FAKE": "Coordinated disinformation campaign",
            "MISLEADING": "Narrative manipulation cluster",
            "UNVERIFIED": "Emerging information cluster",
            "TRUE": "Verified information cluster",
        }
        narrative_family = families.get(most_common_verdict, "Unknown cluster")
    else:
        narrative_family = "Isolated claim — no similar narratives found"

    cluster_id = f"cluster_{abs(hash(claim)) % 100000}" if similar else None

    return NetworkAnalysis(
        claim=claim,
        similar_claims=similar,
        cluster_id=cluster_id,
        narrative_family=narrative_family,
    )


@router.get("/stats", summary="📊 System statistics")
def get_stats():
    """Returns overall stats: total claims indexed, verdicts distribution, top categories."""
    try:
        client = qs._get_client()
        claims_count = client.get_collection("claims").points_count
        sources_count = client.get_collection("sources").points_count

        # Scroll to get verdict distribution
        results, _ = client.scroll(collection_name="claims", limit=500, with_payload=True, with_vectors=False)
        from collections import Counter
        verdicts = Counter(r.payload.get("verdict", "UNKNOWN") for r in results)
        categories = Counter(r.payload.get("category", "general") for r in results)

        return {
            "total_claims_indexed": claims_count,
            "total_sources_indexed": sources_count,
            "verdict_distribution": dict(verdicts),
            "top_categories": dict(categories.most_common(5)),
            "qdrant_collections": ["claims", "verdicts", "sources", "trending"],
        }
    except Exception as e:
        return {"error": str(e), "total_claims_indexed": 0}
