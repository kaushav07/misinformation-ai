"""
Counter-narrative generator.

Uses Claude API when configured, else rich rule-based templates.
Generates: summary, inconsistencies, verified_alternative, advice, citations.
"""
from loguru import logger
from app.config import settings
from app.models.schemas import CounterNarrative, FactCheckResult
from app.models.enums import Verdict, Language


# ─── Claude-powered generation ───────────────────────────────────────────────

def _claude_generate(
    claim: str,
    verdict: Verdict,
    fact_result: FactCheckResult,
    target_language: Language,
) -> CounterNarrative:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

        lang_instruction = {
            Language.HINDI: "Respond entirely in Hindi (Devanagari script).",
            Language.TAMIL: "Respond entirely in Tamil script.",
            Language.ENGLISH: "Respond in clear, simple English.",
        }.get(target_language, "Respond in English.")

        sources_text = "\n".join(f"- {s}" for s in fact_result.sources) if fact_result.sources else "- General knowledge"
        explanation = fact_result.explanation or "No additional context available."

        system_prompt = (
            "You are a fact-checking assistant. Generate accurate, non-partisan counter-narratives. "
            "Return ONLY valid JSON with keys: summary, inconsistencies (list of strings), "
            "verified_alternative, advice, citations (list of strings). No markdown, no explanation."
        )
        user_prompt = (
            f"Claim: \"{claim}\"\n"
            f"Verdict: {verdict.value}\n"
            f"Confidence: {fact_result.confidence:.0%}\n"
            f"Explanation: {explanation}\n"
            f"Sources:\n{sources_text}\n\n"
            f"{lang_instruction}\n\n"
            "Generate a counter-narrative JSON."
        )

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content": user_prompt}],
            system=system_prompt,
        )
        import json
        raw = message.content[0].text.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)

        return CounterNarrative(
            summary=data.get("summary", ""),
            inconsistencies=data.get("inconsistencies", []),
            verified_alternative=data.get("verified_alternative", ""),
            advice=data.get("advice", ""),
            citations=data.get("citations", []),
        )
    except Exception as e:
        logger.warning(f"Claude generation failed: {e}. Using rule-based fallback.")
        return _rule_based_generate(claim, verdict, fact_result, target_language)


# ─── Rule-based fallback with explanation-aware templates ────────────────────

def _rule_based_generate(
    claim: str,
    verdict: Verdict,
    fact_result: FactCheckResult,
    target_language: Language,
) -> CounterNarrative:
    """
    Uses the fact_result.explanation (which now contains real content from the
    knowledge base or NLI) to build a specific counter-narrative.
    """
    explanation = fact_result.explanation or ""
    sources = fact_result.sources[:3] if fact_result.sources else []
    citations = [s for s in sources if "Knowledge Base" not in s and "NLI" not in s] or sources

    if verdict == Verdict.TRUE:
        summary = explanation if explanation else "This claim appears to be accurate based on available evidence."
        return CounterNarrative(
            summary=summary,
            inconsistencies=[],
            verified_alternative="The information in this claim is supported by credible sources.",
            advice="This appears to be reliable information. Always verify important claims through multiple trusted sources.",
            citations=citations,
        )

    if verdict == Verdict.FAKE:
        # Use the explanation from the knowledge base / NLI as the summary
        summary = explanation if explanation else (
            "This claim has been identified as false based on multiple verified sources and fact-checking analysis."
        )
        # Build specific inconsistencies from the explanation
        inconsistencies = [
            f"The claim directly contradicts verified facts: {explanation[:200]}" if explanation else
            "The claim contradicts well-established scientific or factual consensus.",
            "No credible sources or peer-reviewed studies support this statement.",
            "The claim uses emotionally charged language designed to bypass critical thinking.",
        ]
        if target_language == Language.HINDI:
            summary = (
                "यह दावा गलत है। " + explanation
                if explanation else
                "यह दावा कई विश्वसनीय स्रोतों के आधार पर गलत पाया गया है।"
            )
            inconsistencies = [
                "यह दावा स्थापित तथ्यों के विरुद्ध है।",
                "किसी विश्वसनीय स्रोत ने इसका समर्थन नहीं किया है।",
                "इस दावे में भ्रामक भाषा का उपयोग किया गया है।",
            ]
            return CounterNarrative(
                summary=summary,
                inconsistencies=inconsistencies,
                verified_alternative="सही जानकारी के लिए विश्वसनीय स्रोतों से जाँच करें।",
                advice="किसी भी जानकारी को साझा करने से पहले Alt News, Boom Live, या PIB Fact Check से जाँचें।",
                citations=citations,
            )
        return CounterNarrative(
            summary=summary,
            inconsistencies=inconsistencies,
            verified_alternative=f"The correct information: {explanation}" if explanation else
                "The evidence from credible sources contradicts this claim entirely.",
            advice="Before sharing, verify through trusted fact-checking platforms: Alt News, Boom Live, PIB Fact Check, Snopes, or official government websites.",
            citations=citations,
        )

    if verdict == Verdict.MISLEADING:
        summary = (
            f"This claim is partially true but misleading. {explanation}"
            if explanation else
            "This claim contains partial truth but is presented in a misleading way that distorts the full picture."
        )
        if target_language == Language.HINDI:
            return CounterNarrative(
                summary="यह दावा आंशिक रूप से सच है लेकिन भ्रामक तरीके से प्रस्तुत किया गया है।",
                inconsistencies=["दावे को संदर्भ से बाहर लिया गया है।", "महत्वपूर्ण जानकारी को छोड़ दिया गया है।"],
                verified_alternative="पूरा संदर्भ जानने के लिए मूल स्रोत पढ़ें।",
                advice="किसी भी आँकड़े या उद्धरण का मूल स्रोत जाँचें।",
                citations=citations,
            )
        return CounterNarrative(
            summary=summary,
            inconsistencies=[
                "The claim takes facts out of context to support a misleading narrative.",
                "Important context or counter-evidence has been deliberately omitted.",
                "Statistical data or quotes, if used, may be selectively chosen.",
            ],
            verified_alternative="Reading the full context from primary sources is essential to understand the complete picture.",
            advice="Look for the original source of any statistics or quotes, and check whether surrounding context changes the meaning significantly.",
            citations=citations,
        )

    if verdict == Verdict.SATIRE:
        return CounterNarrative(
            summary="This content appears to be satire or parody, not factual reporting.",
            inconsistencies=[
                "The content uses exaggeration and irony typical of satire.",
                "This may have been shared without its satirical context.",
            ],
            verified_alternative="This is satirical content. The events described did not necessarily occur.",
            advice="Check the original source — satire sites often have disclaimers. Do not share satirical content as factual news.",
            citations=citations,
        )

    # UNVERIFIED
    return CounterNarrative(
        summary=f"This claim could not be fully verified. {explanation}" if explanation else
            "This claim could not be verified against available data. Insufficient credible evidence exists to confirm or deny it.",
        inconsistencies=[
            "Insufficient credible sources exist to support or refute this claim.",
            "The claim may be too recent or too niche for existing fact-checking databases.",
        ],
        verified_alternative="No strong evidence was found to confirm or deny this claim. Treat with caution.",
        advice="Exercise caution with unverified information. Wait for credible reporting before acting on or sharing this claim.",
        citations=citations,
    )


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_counter_narrative(
    claim: str,
    verdict: Verdict,
    fact_result: FactCheckResult,
    target_language: Language = Language.ENGLISH,
) -> CounterNarrative:
    api_key = settings.anthropic_api_key
    use_claude = bool(api_key and api_key != "your_anthropic_api_key_here")

    if use_claude:
        return _claude_generate(claim, verdict, fact_result, target_language)
    return _rule_based_generate(claim, verdict, fact_result, target_language)