"""
Initialise all Qdrant collections for the project.

Run ONCE before starting the server:
    cd backend && python ../qdrant/init_collections.py

Collections created:
    claims    — verified claims cache (main semantic dedup)
    verdicts  — detailed verdicts with metadata
    sources   — indexed credible news sources
    trending  — high-frequency misinformation tracker
"""
import sys
import os

# Allow imports from backend/app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.services.qdrant_service import init_all_collections, _get_client
from loguru import logger


def verify_collections():
    client = _get_client()
    collections = client.get_collections().collections
    print("\n✅ Qdrant collections:")
    for c in collections:
        info = client.get_collection(c.name)
        print(f"   {c.name:20s} — {info.points_count} points")
    print()


if __name__ == "__main__":
    print("🔧 Initialising Qdrant collections…")
    init_all_collections()
    verify_collections()
    print("🚀 Ready to run the API!")
