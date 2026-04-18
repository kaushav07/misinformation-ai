"""
Qdrant vector database service.

Collections:
  claims     — individual verified claims (main dedup cache)
  verdicts   — fact-check verdicts with metadata
  sources    — indexed trusted sources / articles

All operations use paraphrase-multilingual-MiniLM-L12-v2 (dim=384)
which handles Hindi, Tamil, and English natively.
"""
import uuid
from typing import Optional, List
from loguru import logger
from app.config import settings
from app.models.enums import Verdict

# ─── Clients (lazy init) ─────────────────────────────────────────────────────

_qdrant_client = None
_embed_model = None

COLLECTIONS = {
    "claims": 384,
    "verdicts": 384,
    "sources": 384,
    "trending": 384,
}

SIMILARITY_THRESHOLD = 0.85


def _get_client():
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        logger.info("Connecting to Qdrant…")
        _qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=15,
        )
        logger.info("Qdrant connected ✅")
    return _qdrant_client


def _get_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model…")
        _embed_model = SentenceTransformer(settings.embed_model)
        logger.info("Embedding model loaded ✅")
    return _embed_model


def _embed(text: str) -> List[float]:
    model = _get_model()
    return model.encode(text, normalize_embeddings=True).tolist()


# ─── Collection management ────────────────────────────────────────────────────

def init_all_collections():
    """Create all collections if they don't already exist."""
    from qdrant_client.models import VectorParams, Distance
    client = _get_client()
    existing = {c.name for c in client.get_collections().collections}

    for name, dim in COLLECTIONS.items():
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info(f"Collection '{name}' created ✅")
        else:
            logger.info(f"Collection '{name}' already exists.")


def create_collection():
    """Backward-compat wrapper."""
    init_all_collections()


# ─── Claims collection ────────────────────────────────────────────────────────

def upsert_claim(claim: str, verdict: str, risk_score: int, sources: List[str] = None):
    """Store a verified claim in the claims collection."""
    from qdrant_client.models import PointStruct
    client = _get_client()
    vector = _embed(claim)

    client.upsert(
        collection_name="claims",
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "claim": claim,
                    "verdict": verdict,
                    "risk_score": risk_score,
                    "sources": sources or [],
                    "hit_count": 1,
                },
            )
        ],
    )
    logger.debug(f"Upserted claim: '{claim[:50]}' | verdict={verdict}")


def search_similar_claim(claim: str) -> Optional[dict]:
    """
    Search for a semantically similar claim already in the cache.
    Returns payload dict if similarity > threshold, else None.
    """
    client = _get_client()
    vector = _embed(claim)

    try:
        results = client.query_points(
            collection_name="claims",
            query=vector,
            limit=1,
            with_payload=True,
        )

        if results.points:
            point = results.points[0]
            score = point.score
            payload = point.payload

            if score >= SIMILARITY_THRESHOLD:
                logger.info(f"Cache HIT (score={score:.3f}): '{claim[:50]}'")
                # Increment hit_count
                try:
                    client.set_payload(
                        collection_name="claims",
                        payload={"hit_count": payload.get("hit_count", 1) + 1},
                        points=[point.id],
                    )
                except Exception:
                    pass
                return payload

        logger.debug(f"Cache MISS: '{claim[:50]}'")
        return None

    except Exception as e:
        logger.warning(f"Qdrant search failed: {e}")
        return None


def get_similar_claims(claim: str, top_k: int = 5) -> List[dict]:
    """Get top-k similar claims for network/cluster analysis."""
    client = _get_client()
    vector = _embed(claim)

    try:
        results = client.query_points(
            collection_name="claims",
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            {**p.payload, "similarity": p.score}
            for p in results.points
            if p.score >= 0.60  # broader threshold for network view
        ]
    except Exception as e:
        logger.warning(f"Similar claims search failed: {e}")
        return []


def get_trending_claims(limit: int = 10) -> List[dict]:
    """
    Return top claims by hit_count from the claims collection.
    UNIQUE FEATURE: Trending misinformation tracker.
    """
    from qdrant_client.models import Filter, FieldCondition, Range
    client = _get_client()

    try:
        # Scroll through all points and sort by hit_count
        results, _ = client.scroll(
            collection_name="claims",
            limit=200,
            with_payload=True,
            with_vectors=False,
        )

        sorted_claims = sorted(
            results,
            key=lambda p: p.payload.get("hit_count", 0),
            reverse=True,
        )

        return [
            {
                **p.payload,
                "id": str(p.id),
            }
            for p in sorted_claims[:limit]
        ]

    except Exception as e:
        logger.warning(f"Trending claims fetch failed: {e}")
        return []


# ─── Sources collection ───────────────────────────────────────────────────────

def index_source(title: str, content: str, url: str, credibility: float):
    """Index a news article / source for reference retrieval."""
    from qdrant_client.models import PointStruct
    client = _get_client()
    vector = _embed(f"{title} {content[:200]}")

    client.upsert(
        collection_name="sources",
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "title": title,
                    "content": content[:500],
                    "url": url,
                    "credibility": credibility,
                },
            )
        ],
    )


def search_relevant_sources(claim: str, top_k: int = 3) -> List[dict]:
    """Retrieve most relevant trusted sources for a claim."""
    client = _get_client()
    vector = _embed(claim)

    try:
        results = client.query_points(
            collection_name="sources",
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        return [p.payload for p in results.points]
    except Exception as e:
        logger.warning(f"Source search failed: {e}")
        return []


# ─── Virality prediction (UNIQUE FEATURE) ────────────────────────────────────

def predict_virality(claim: str, verdict: str, risk_score: int) -> dict:
    """
    Unique feature: predict how viral a misinformation claim might go.
    Uses cluster size (similar claims in DB) + risk score + verdict.
    """
    similar = get_similar_claims(claim, top_k=20)
    cluster_size = len(similar)

    # Base virality from risk
    base = risk_score / 100.0

    # Amplify by cluster size (many similar = already spreading)
    cluster_factor = min(1.0, cluster_size / 10.0)

    # Verdict amplification
    verdict_amp = {
        "FAKE": 1.3,
        "MISLEADING": 1.15,
        "UNVERIFIED": 1.0,
        "TRUE": 0.7,
    }.get(verdict, 1.0)

    virality_score = min(1.0, base * verdict_amp * (1 + cluster_factor * 0.5))

    if virality_score >= 0.75:
        predicted_reach = "Millions (high viral potential)"
        spread_velocity = "Very fast — could spread within hours"
    elif virality_score >= 0.50:
        predicted_reach = "Hundreds of thousands"
        spread_velocity = "Moderate — likely within 1-3 days"
    elif virality_score >= 0.25:
        predicted_reach = "Tens of thousands"
        spread_velocity = "Slow — limited spread expected"
    else:
        predicted_reach = "Limited (< 10,000)"
        spread_velocity = "Very slow — unlikely to go viral"

    return {
        "virality_score": round(virality_score, 3),
        "predicted_reach": predicted_reach,
        "spread_velocity": spread_velocity,
        "cluster_size": cluster_size,
        "risk_amplification": round(verdict_amp, 2),
    }
