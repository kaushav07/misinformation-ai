"""
Seed Qdrant with known Indian misinformation claims.
Provides a warm cache from Day 1 for better demo results.

Run after init_collections.py:
    cd backend && python ../qdrant/seed_data.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.services.qdrant_service import upsert_claim, index_source
from loguru import logger

# ─── Known misinformation claims (pre-seeded) ─────────────────────────────────
SEED_CLAIMS = [
    # Health misinformation
    {
        "claim": "5G towers spread COVID-19 by activating the virus through electromagnetic radiation",
        "verdict": "FAKE",
        "risk_score": 94,
        "sources": ["WHO", "CDC", "Full Fact"],
    },
    {
        "claim": "COVID-19 vaccines contain microchips to track people using 5G networks",
        "verdict": "FAKE",
        "risk_score": 97,
        "sources": ["WHO", "Reuters Fact Check", "Snopes"],
    },
    {
        "claim": "Drinking cow urine cures coronavirus",
        "verdict": "FAKE",
        "risk_score": 95,
        "sources": ["ICMR", "WHO", "The Wire Science"],
    },
    {
        "claim": "Onions placed in rooms absorb viruses and protect against flu",
        "verdict": "FAKE",
        "risk_score": 72,
        "sources": ["CDC", "NHS"],
    },
    {
        "claim": "Homeopathy can cure cancer completely without any side effects",
        "verdict": "FAKE",
        "risk_score": 88,
        "sources": ["ICMR", "AIIMS", "Lancet Oncology"],
    },
    # Political misinformation
    {
        "claim": "The Indian government has declared a national emergency and suspended the constitution",
        "verdict": "FAKE",
        "risk_score": 91,
        "sources": ["PIB Fact Check", "Press Trust of India"],
    },
    {
        "claim": "Electronic voting machines (EVMs) in India are pre-programmed to favour the ruling party",
        "verdict": "MISLEADING",
        "risk_score": 68,
        "sources": ["Election Commission of India", "Supreme Court of India"],
    },
    # Environmental
    {
        "claim": "Climate change is a hoax invented by scientists to secure research funding",
        "verdict": "FAKE",
        "risk_score": 89,
        "sources": ["NASA", "IPCC", "Nature Climate Change"],
    },
    # Financial
    {
        "claim": "RBI has announced that all Rs 500 notes will be invalid from next month",
        "verdict": "FAKE",
        "risk_score": 87,
        "sources": ["RBI", "PIB Fact Check"],
    },
    # Satire/misleading
    {
        "claim": "Scientists have discovered that reading news causes 40% drop in IQ within weeks",
        "verdict": "FAKE",
        "risk_score": 75,
        "sources": ["No credible source found — likely satire or fabricated"],
    },
    # Hindi misinformation (stored in English after translation)
    {
        "claim": "The government will ban WhatsApp in India next week due to security concerns",
        "verdict": "FAKE",
        "risk_score": 82,
        "sources": ["Ministry of Electronics and IT", "PIB Fact Check"],
    },
    {
        "claim": "Eating turmeric with hot milk every morning completely cures diabetes",
        "verdict": "MISLEADING",
        "risk_score": 61,
        "sources": ["AIIMS", "Indian Journal of Medical Research"],
    },
    # Verified true claims (for balance)
    {
        "claim": "India launched Chandrayaan-3 mission to the Moon in 2023",
        "verdict": "TRUE",
        "risk_score": 5,
        "sources": ["ISRO", "Times of India", "BBC"],
    },
    {
        "claim": "The COVID-19 pandemic was declared a public health emergency by WHO in January 2020",
        "verdict": "TRUE",
        "risk_score": 4,
        "sources": ["WHO", "Reuters"],
    },
]

# ─── Trusted sources to index ─────────────────────────────────────────────────
SEED_SOURCES = [
    {
        "title": "PIB Fact Check Portal",
        "content": "Official fact-checking initiative by the Press Information Bureau, Government of India.",
        "url": "https://pib.gov.in/factcheck.aspx",
        "credibility": 0.97,
    },
    {
        "title": "BOOM Live — Indian Fact Checker",
        "content": "BOOM is an independent digital journalism initiative that produces fact checks on viral misinformation.",
        "url": "https://www.boomlive.in",
        "credibility": 0.93,
    },
    {
        "title": "Alt News",
        "content": "Alt News is an Indian fact-checking website that verifies claims circulating on social media.",
        "url": "https://www.altnews.in",
        "credibility": 0.92,
    },
    {
        "title": "WHO Health Claims Database",
        "content": "World Health Organization official health information and myth busting resources.",
        "url": "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/myth-busters",
        "credibility": 0.98,
    },
    {
        "title": "Reuters Fact Check",
        "content": "Reuters fact-checking team verifying viral claims globally.",
        "url": "https://www.reuters.com/fact-check",
        "credibility": 0.96,
    },
]


def seed_all():
    print("🌱 Seeding Qdrant with known claims…")
    for i, item in enumerate(SEED_CLAIMS):
        try:
            upsert_claim(
                claim=item["claim"],
                verdict=item["verdict"],
                risk_score=item["risk_score"],
                sources=item["sources"],
            )
            print(f"  [{i+1:2d}/{len(SEED_CLAIMS)}] ✅ {item['claim'][:60]}…")
        except Exception as e:
            print(f"  [{i+1:2d}/{len(SEED_CLAIMS)}] ❌ Failed: {e}")

    print("\n🌱 Indexing trusted sources…")
    for src in SEED_SOURCES:
        try:
            index_source(src["title"], src["content"], src["url"], src["credibility"])
            print(f"  ✅ {src['title']}")
        except Exception as e:
            print(f"  ❌ {src['title']}: {e}")

    print(f"\n✅ Seeding complete — {len(SEED_CLAIMS)} claims, {len(SEED_SOURCES)} sources indexed.")


if __name__ == "__main__":
    seed_all()
