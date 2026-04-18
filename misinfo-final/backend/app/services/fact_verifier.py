"""
Fact verification pipeline.

NOTE: Wikipedia and external search are NOT available in this deployment
(network allowlist blocks them). The pipeline runs in KB + NLI-only mode.

Strategy:
  1. Knowledge base  — instant, highest accuracy for known claims
  2. Claim-type heuristics — pattern-match claim structure for risk signals
  3. Targeted NLI  — domain-specific hypotheses per claim category
  4. Calibrated decision — lower thresholds + type-aware boosting
  5. Google FC API  — used if key configured (optional)
  6. Serper search  — used if key configured (optional)
"""
import re
import httpx
from typing import Optional, List, Tuple
from loguru import logger
from app.config import settings
from app.models.schemas import FactCheckResult
from app.models.enums import Verdict


# ─── NLI model ───────────────────────────────────────────────────────────────

_nli_pipeline = None

def _get_nli():
    global _nli_pipeline
    if _nli_pipeline is None:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading NLI model…")
        _nli_pipeline = hf_pipeline(
            "zero-shot-classification",
            model=settings.nli_model,
            device=-1,
        )
        logger.info("NLI model loaded ✅")
    return _nli_pipeline


# ─── Knowledge base ──────────────────────────────────────────────────────────

_KNOWN_FACTS = [
    # ── Geography — India ──────────────────────────────────────────────────
    (r"delhi\s+is\s+(?:the\s+)?capital\s+of\s+(?:india|bharat)", Verdict.TRUE, 0.99,
     "New Delhi is the capital of India. This is correct."),
    (r"(?:new\s+delhi|delhi)\s+is\s+(?:the\s+)?capital\s+of\s+india", Verdict.TRUE, 0.99,
     "New Delhi is the capital of India. This is correct."),
    (r"(?:mumbai|bombay)\s+is\s+(?:the\s+)?capital\s+of\s+(?:india|bharat)", Verdict.FAKE, 0.99,
     "Mumbai is NOT the capital of India. New Delhi is. Mumbai is the financial capital."),
    (r"delhi\s+is\s+(?:the\s+)?capital\s+of\s+mumbai", Verdict.FAKE, 0.99,
     "Delhi is not the capital of Mumbai. Delhi is India's national capital; Mumbai is a city in Maharashtra."),
    (r"(?:mumbai|bombay)\s+is\s+(?:the\s+)?capital\s+of\s+maharashtra", Verdict.TRUE, 0.97,
     "Mumbai is the capital of Maharashtra. This is correct."),
    (r"(?:chennai|madras)\s+is\s+(?:the\s+)?capital\s+of\s+tamil\s*nadu", Verdict.TRUE, 0.97,
     "Chennai is the capital of Tamil Nadu. This is correct."),
    (r"(?:kolkata|calcutta)\s+is\s+(?:the\s+)?capital\s+of\s+west\s*bengal", Verdict.TRUE, 0.97,
     "Kolkata is the capital of West Bengal. This is correct."),
    (r"(?:bangalore|bengaluru)\s+is\s+(?:the\s+)?capital\s+of\s+karnataka", Verdict.TRUE, 0.97,
     "Bengaluru (Bangalore) is the capital of Karnataka. This is correct."),
    (r"hyderabad\s+is\s+(?:the\s+)?capital\s+of\s+telangana", Verdict.TRUE, 0.97,
     "Hyderabad is the capital of Telangana. This is correct."),
    (r"jaipur\s+is\s+(?:the\s+)?capital\s+of\s+rajasthan", Verdict.TRUE, 0.97,
     "Jaipur is the capital of Rajasthan. This is correct."),
    (r"lucknow\s+is\s+(?:the\s+)?capital\s+of\s+uttar\s*pradesh", Verdict.TRUE, 0.97,
     "Lucknow is the capital of Uttar Pradesh. This is correct."),
    (r"patna\s+is\s+(?:the\s+)?capital\s+of\s+bihar", Verdict.TRUE, 0.97,
     "Patna is the capital of Bihar. This is correct."),
    (r"bhopal\s+is\s+(?:the\s+)?capital\s+of\s+madhya\s*pradesh", Verdict.TRUE, 0.97,
     "Bhopal is the capital of Madhya Pradesh. This is correct."),
    (r"ahmedabad\s+is\s+(?:the\s+)?capital\s+of\s+gujarat", Verdict.MISLEADING, 0.85,
     "Gandhinagar, not Ahmedabad, is the capital of Gujarat. Ahmedabad is the largest city."),
    (r"gandhinagar\s+is\s+(?:the\s+)?capital\s+of\s+gujarat", Verdict.TRUE, 0.97,
     "Gandhinagar is the capital of Gujarat. This is correct."),

    # ── Geography — World ──────────────────────────────────────────────────
    (r"paris\s+is\s+(?:the\s+)?capital\s+of\s+france", Verdict.TRUE, 0.99,
     "Paris is the capital of France. This is correct."),
    (r"london\s+is\s+(?:the\s+)?capital\s+of\s+(?:uk|united\s+kingdom|england|britain)", Verdict.TRUE, 0.99,
     "London is the capital of the United Kingdom. This is correct."),
    (r"washington\s+(?:d\.?c\.?)?\s+is\s+(?:the\s+)?capital\s+of\s+(?:usa|united\s+states|america)", Verdict.TRUE, 0.99,
     "Washington D.C. is the capital of the United States. This is correct."),
    (r"beijing\s+is\s+(?:the\s+)?capital\s+of\s+china", Verdict.TRUE, 0.99,
     "Beijing is the capital of China. This is correct."),
    (r"moscow\s+is\s+(?:the\s+)?capital\s+of\s+russia", Verdict.TRUE, 0.99,
     "Moscow is the capital of Russia. This is correct."),
    (r"islamabad\s+is\s+(?:the\s+)?capital\s+of\s+pakistan", Verdict.TRUE, 0.99,
     "Islamabad is the capital of Pakistan. This is correct."),
    (r"dhaka\s+is\s+(?:the\s+)?capital\s+of\s+bangladesh", Verdict.TRUE, 0.99,
     "Dhaka is the capital of Bangladesh. This is correct."),
    (r"kathmandu\s+is\s+(?:the\s+)?capital\s+of\s+nepal", Verdict.TRUE, 0.99,
     "Kathmandu is the capital of Nepal. This is correct."),
    (r"canberra\s+is\s+(?:the\s+)?capital\s+of\s+australia", Verdict.TRUE, 0.99,
     "Canberra is the capital of Australia. This is correct."),
    (r"sydney\s+is\s+(?:the\s+)?capital\s+of\s+australia", Verdict.FAKE, 0.97,
     "Sydney is NOT the capital of Australia. Canberra is."),
    (r"toronto\s+is\s+(?:the\s+)?capital\s+of\s+canada", Verdict.FAKE, 0.97,
     "Toronto is NOT the capital of Canada. Ottawa is."),

    # ── Science — debunked claims ──────────────────────────────────────────
    (r"earth\s+is\s+flat", Verdict.FAKE, 0.99,
     "The Earth is not flat. It is an oblate spheroid, proven by satellite imagery, physics, and centuries of observation."),
    (r"earth\s+is\s+(?:the\s+)?cent(?:er|re)\s+of\s+(?:the\s+)?(?:universe|solar\s*system)", Verdict.FAKE, 0.99,
     "The Earth is not the center of the universe or solar system. The Sun is the centre of our solar system."),
    (r"(?:humans?|people)\s+(?:evolved?|descend(?:ed)?)\s+from\s+(?:monkeys|apes)", Verdict.MISLEADING, 0.88,
     "Humans did not evolve FROM apes. Both humans and apes share a common ancestor — this is a common misstatement of evolutionary theory."),
    (r"vaccines?\s+(?:cause[sd]?|causes?|lead\s+to)\s+autism", Verdict.FAKE, 0.99,
     "Vaccines do not cause autism. The 1998 study claiming this was retracted and its author struck off. No credible scientific evidence supports this."),
    (r"(?:5g|5\s*g)\s+(?:towers?|network|radiation)?\s*(?:spread|cause[sd]?|transmit|give[s]?)\s+(?:covid|corona|cancer|virus)", Verdict.FAKE, 0.99,
     "5G cannot spread COVID-19 or cause cancer. Viruses do not travel on radio waves. This is a thoroughly debunked conspiracy theory."),
    (r"(?:covid|corona).*(?:microchip|chip|track|nanotechnology|nano\s*tech)", Verdict.FAKE, 0.99,
     "COVID-19 vaccines do not contain microchips or nanotechnology tracking devices. This is a debunked conspiracy theory."),
    (r"(?:drinking|consume?|consuming)\s+(?:cow\s+urine|bleach|dettol|disinfectant)\s+(?:cure[sd]?|treat[s]?|kill[s]?|prevent[s]?)\s+(?:covid|corona|virus|disease)", Verdict.FAKE, 0.99,
     "Drinking cow urine, bleach, or disinfectants does NOT cure COVID-19 and is extremely dangerous."),
    (r"sun\s+revolves?\s+around\s+(?:the\s+)?earth", Verdict.FAKE, 0.99,
     "The Earth revolves around the Sun, not the other way around."),
    (r"climate\s+change\s+is\s+(?:a\s+)?(?:hoax|fake|fabricated|invented|lie|conspiracy)", Verdict.FAKE, 0.97,
     "Climate change is not a hoax. It is supported by overwhelming scientific consensus from NASA, IPCC, and virtually every major scientific institution."),
    (r"global\s+warming\s+is\s+(?:a\s+)?(?:hoax|fake|lie|not\s+real)", Verdict.FAKE, 0.97,
     "Global warming is real and supported by decades of empirical evidence across multiple independent scientific disciplines."),

    # ── Currency / financial conspiracy ────────────────────────────────────
    (r"(?:government|govt|rbi|modi)\s+(?:embedded?|implanted?|put|placed?|inserted?)\s+"
     r"(?:tracking\s+)?(?:chips?|microchips?|gps|sensors?)\s+in\s+(?:new\s+)?(?:currency|notes?|rupees?|money)", Verdict.FAKE, 0.99,
     "Indian currency notes do NOT contain tracking chips, GPS, or microchips. This is a viral conspiracy theory with no factual basis. "
     "RBI and the government have never embedded electronic tracking devices in banknotes."),
    (r"(?:new|2000|500|200|100)\s*(?:rupee\s+)?notes?\s+(?:contain|have|embedded?)\s+"
     r"(?:tracking\s+)?(?:chips?|microchips?|gps|sensors?|nano)", Verdict.FAKE, 0.98,
     "Indian currency notes do not contain tracking chips, GPS, or nanotechnology. This conspiracy theory has been debunked by the RBI and fact-checkers."),
    (r"(?:rupee|inr|indian\s+currency)\s+(?:has\s+)?(?:collapsed|crashed|become\s+worthless|demonetis|worthless)", Verdict.MISLEADING, 0.80,
     "The Indian Rupee fluctuates but has not collapsed. For current exchange rates, check RBI or financial news sources."),
    (r"(?:rbi|government)\s+(?:is\s+)?(?:banning|will\s+ban|going\s+to\s+ban)\s+(?:all\s+)?(?:cash|currency|rupee)", Verdict.FAKE, 0.90,
     "The RBI has not announced any ban on cash or the Indian Rupee. Verify through official RBI or government announcements."),

    # ── Tech / social media hoaxes ─────────────────────────────────────────
    (r"(?:whatsapp|facebook|instagram|twitter|youtube)\s+(?:is\s+)?(?:shutting\s+down|banned|closing|going\s+to\s+(?:shut|close|end))", Verdict.FAKE, 0.95,
     "This is a recurring viral hoax. These platforms have not announced shutdown or been banned. Verify through official company sources."),
    (r"whatsapp\s+(?:will\s+)?(?:start\s+)?(?:charging|charge)\s+(?:fees?|money|rupees?)", Verdict.FAKE, 0.93,
     "WhatsApp remains free to use. This 'WhatsApp will charge fees' message is a recurring hoax that has circulated since 2012."),
    (r"forward\s+this\s+(?:message|post)\s+to\s+\d+\s+(?:people|contacts|friends)", Verdict.FAKE, 0.88,
     "Chain messages asking you to forward to N contacts are almost always hoaxes or scams. No platform rewards or penalises forwarding behaviour."),
    (r"whatsapp\s+(?:is\s+)?(?:going\s+to\s+)?(?:delete|deactivate)\s+(?:inactive\s+)?accounts?", Verdict.MISLEADING, 0.80,
     "WhatsApp does deactivate accounts after prolonged inactivity, but the specific timelines in viral messages are often exaggerated or fabricated."),

    # ── Government / political hoaxes — India ─────────────────────────────
    (r"(?:modi|government|govt|india)\s+(?:has\s+)?(?:declared|imposed?|announce[sd]?)\s+(?:a\s+)?(?:national\s+)?emergency", Verdict.FAKE, 0.92,
     "No national emergency has been declared in India. Verify through official PIB or government channels before sharing."),
    (r"(?:free|complimentary)\s+(?:electricity|internet|data|petrol|gas|lpg|ration)\s+(?:for\s+all|scheme|yojana)", Verdict.MISLEADING, 0.82,
     "Government subsidy schemes exist but viral messages often exaggerate or fabricate benefits. Verify through official government portals (india.gov.in or PIB)."),
    (r"(?:mahatma\s+)?gandhi\s+(?:was|is)\s+(?:the\s+)?(?:first\s+)?prime\s+minister\s+of\s+india", Verdict.FAKE, 0.99,
     "Mahatma Gandhi was never the Prime Minister of India. Jawaharlal Nehru was India's first PM (1947–1964)."),
    (r"jawaharlal\s+nehru\s+was\s+(?:the\s+)?first\s+prime\s+minister\s+of\s+india", Verdict.TRUE, 0.99,
     "Jawaharlal Nehru was India's first Prime Minister (1947–1964). This is correct."),
    (r"(?:evm|electronic\s+voting\s+machine)\s+(?:is|are)\s+(?:hacked|rigged|pre.?programmed|manipulated)", Verdict.MISLEADING, 0.78,
     "EVM tampering allegations are a contested political claim. The Supreme Court of India and Election Commission have repeatedly upheld EVM integrity. No proven instance of EVM hacking has been established in court."),
    (r"india\s+(?:got|gained|received)\s+independence\s+in\s+1947", Verdict.TRUE, 0.99,
     "India gained independence on 15 August 1947. This is correct."),
    (r"india\s+(?:got|gained|received)\s+independence\s+in\s+(?:1945|1946|1948|1949|1950|1952)", Verdict.FAKE, 0.99,
     "India gained independence on 15 August 1947, not the year mentioned."),
    (r"(?:india|constitution)\s+(?:was\s+)?(?:a\s+)?republic\s+(?:since|from|on)\s+(?:january\s+)?(?:26|1950)", Verdict.TRUE, 0.97,
     "India became a Republic on 26 January 1950 when the Constitution came into force. This is correct."),

    # ── Health hoaxes ──────────────────────────────────────────────────────
    (r"(?:ayurveda|turmeric|haldi|neem|giloy|ashwagandha)\s+(?:cures?|treats?|prevents?)\s+(?:cancer|diabetes|covid|hiv|aids|tumor)", Verdict.FAKE, 0.92,
     "No herbal remedy cures cancer, diabetes, HIV, or COVID-19. These conditions require treatment from qualified medical professionals."),
    (r"(?:drinking|eating)\s+(?:onion|garlic|ginger|lemon)\s+(?:juice\s+)?(?:prevents?|cures?|kills?)\s+(?:covid|corona|virus|cancer)", Verdict.FAKE, 0.90,
     "No food item prevents or cures COVID-19 or cancer. Eat well for general health but do not rely on food as medicine for serious diseases."),
    (r"(?:hot\s+water|steam|salt\s+water)\s+(?:kills?|prevents?|cures?|destroys?)\s+(?:covid|corona|coronavirus)", Verdict.FAKE, 0.92,
     "Hot water, steam, and salt water do not kill the COVID-19 virus inside the body. This is a debunked claim."),
    (r"(?:masks?|face\s+mask)\s+(?:cause[sd]?|causes?|lead\s+to)\s+(?:oxygen\s+deprivation|co2\s+poisoning|hypoxia|brain\s+damage)", Verdict.FAKE, 0.94,
     "Properly worn face masks do not cause oxygen deprivation or CO2 poisoning. Medical professionals wear them for hours safely."),
    (r"blood\s+type\s+(?:o|a|b|ab)\s+(?:is\s+)?(?:immune|protected|resistant)\s+(?:to\s+)?(?:covid|coronavirus)", Verdict.MISLEADING, 0.83,
     "No blood type confers immunity to COVID-19. Some studies suggested minor statistical correlations, but no blood type makes a person immune."),

    # ── Space & history ────────────────────────────────────────────────────
    (r"(?:chandrayaan|chandrayan)-?3\s+(?:successfully\s+)?(?:landed|touchdown)", Verdict.TRUE, 0.99,
     "Chandrayaan-3 successfully landed near the Moon's south pole on 23 August 2023. This is correct."),
    (r"(?:isro|india)\s+(?:never\s+)?(?:landed|reached)\s+(?:the\s+)?moon", Verdict.FAKE, 0.97,
     "India's ISRO successfully landed Chandrayaan-3 near the Moon's south pole on 23 August 2023."),
    (r"(?:nasa|usa|america|apollo)\s+(?:never\s+)?(?:landed|went|went\s+to)\s+(?:the\s+)?moon", Verdict.FAKE, 0.99,
     "NASA landed astronauts on the Moon six times between 1969–1972. The Moon landings are not a hoax."),
    (r"(?:moon\s+landing|apollo\s+11)\s+(?:was\s+)?(?:faked|fake|hoax|staged|fabricated)", Verdict.FAKE, 0.99,
     "The Moon landing was not faked. It is one of the most documented events in history, independently verified by multiple countries including the Soviet Union."),

    # ── Death hoaxes — Indian leaders ─────────────────────────────────────
    (r"(?:narendra\s+)?modi\s+(?:is\s+|has\s+|was\s+)?(?:dead|died|passing|passed\s+away|no\s+more|killed|shot|assassinated)", Verdict.FAKE, 0.97,
     "As of 2025, Narendra Modi is alive and serving as Prime Minister of India. This is a death hoax."),
    (r"(?:narendra\s+)?modi\s+(?:is\s+|has\s+|was\s+)?(?:passed|death)", Verdict.FAKE, 0.97,
     "As of 2025, Narendra Modi is alive. This appears to be a death hoax."),
    (r"(?:amit|amit\s+shah)\s+(?:is\s+|has\s+|was\s+)?(?:dead|died|passed\s+away|killed)", Verdict.FAKE, 0.95,
     "As of 2025, Amit Shah is alive and serving as Home Minister of India."),
    (r"(?:rahul\s+gandhi)\s+(?:is\s+|has\s+|was\s+)?(?:dead|died|passed\s+away|killed)", Verdict.FAKE, 0.95,
     "As of 2025, Rahul Gandhi is alive and active in Indian politics."),
    (r"(?:mamata|mamata\s+banerjee)\s+(?:is\s+|has\s+|was\s+)?(?:dead|died|passed\s+away|killed)", Verdict.FAKE, 0.95,
     "As of 2025, Mamata Banerjee is alive and serving as Chief Minister of West Bengal."),
    (r"(?:arvind\s+)?kejriwal\s+(?:is\s+|has\s+|was\s+)?(?:dead|died|passed\s+away|killed)", Verdict.FAKE, 0.95,
     "As of 2025, Arvind Kejriwal is alive and active in Indian politics."),
    (r"(?:yogi|yogi\s+adityanath)\s+(?:is\s+|has\s+|was\s+)?(?:dead|died|passed\s+away|killed|arrested)", Verdict.FAKE, 0.95,
     "As of 2025, Yogi Adityanath is alive and serving as Chief Minister of Uttar Pradesh."),

    # ── Death hoaxes — global leaders ─────────────────────────────────────
    (r"(?:joe\s+)?biden\s+(?:is\s+|has\s+|was\s+)?(?:dead|died|passed\s+away|killed)", Verdict.FAKE, 0.95,
     "As of 2025, Joe Biden is alive. This appears to be a death hoax."),
    (r"(?:donald\s+)?trump\s+(?:is\s+|has\s+|was\s+)?(?:dead|died|passed\s+away|killed|assassinated)", Verdict.FAKE, 0.95,
     "As of 2025, Donald Trump is alive. This appears to be a death hoax."),
    (r"(?:vladimir\s+)?putin\s+(?:is\s+|has\s+|was\s+)?(?:dead|died|passed\s+away|killed|assassinated)", Verdict.FAKE, 0.95,
     "As of 2025, Vladimir Putin is alive. This appears to be a death hoax."),
    (r"(?:xi\s+jinping|jinping)\s+(?:is\s+|has\s+|was\s+)?(?:dead|died|passed\s+away|killed|ousted|arrested)", Verdict.FAKE, 0.92,
     "As of 2025, Xi Jinping is alive and serving as President of China."),
    (r"(?:elon\s+)?musk\s+(?:is\s+|has\s+|was\s+)?(?:dead|died|passed\s+away|killed|arrested)", Verdict.FAKE, 0.93,
     "As of 2025, Elon Musk is alive."),

    # ── Common WhatsApp / viral forwards ──────────────────────────────────
    (r"(?:government|pm|modi|india)\s+(?:is\s+)?(?:giving|offering|distributing)\s+"
     r"(?:free\s+)?(?:\d+(?:,\d+)*\s+)?(?:rupees?|rs\.?|inr|cash|money)\s+(?:to\s+)?(?:all|every|each)", Verdict.FAKE, 0.95,
     "The government is not distributing free cash to all citizens. This is a common WhatsApp scam designed to steal personal information."),
    (r"(?:click|tap|open)\s+(?:this\s+)?(?:link|url)\s+(?:to\s+)?(?:claim|get|receive)\s+"
     r"(?:free\s+)?(?:money|cash|prize|reward|lottery)", Verdict.FAKE, 0.97,
     "This is a scam. No government or legitimate organisation distributes money through WhatsApp links."),
    (r"congratulations?\s*[,!]?\s*you\s+(?:have\s+)?(?:won|been\s+selected|received)", Verdict.FAKE, 0.92,
     "Unsolicited 'congratulations, you've won' messages are almost always scams. Do not click any links or share personal information."),
    (r"(?:army|military|police|cbi|ib)\s+(?:is\s+)?(?:looking|searching|asking)\s+for\s+"
     r"(?:people|citizens|volunteers)\s+(?:to\s+)?(?:share|forward|spread)", Verdict.FAKE, 0.90,
     "Indian security agencies do not recruit or ask citizens to spread messages via WhatsApp. This is a hoax."),
]


def _knowledge_base_check(claim: str) -> Optional[Tuple[Verdict, float, str]]:
    """Check against hard-coded knowledge. Returns (verdict, confidence, explanation) or None."""
    claim_lower = claim.lower().strip()
    for pattern, verdict, conf, explanation in _KNOWN_FACTS:
        if re.search(pattern, claim_lower, re.IGNORECASE):
            logger.info(f"KB match: '{pattern[:50]}' → {verdict.value}")
            return verdict, conf, explanation
    return None


# ─── Claim-type heuristics ────────────────────────────────────────────────────

# Patterns that are strong signals of misinformation regardless of specific content
_MISINFO_SIGNAL_PATTERNS = [
    # Conspiracy / hidden truth patterns
    (r"(?:secret|hidden|suppressed|they\s+don'?t\s+want\s+you\s+to\s+know|"
     r"exposed|truth\s+about|cover.?up|cover\s+up)", 0.35),
    # "Embedded in / contains X" conspiracy
    (r"(?:embedded?|implanted?|contains?|hidden\s+inside)\s+"
     r"(?:chips?|microchips?|gps|tracking|nano|sensors?|poison)", 0.50),
    # Miracle cure language
    (r"(?:cures?\s+(?:all|every|any)|miracle\s+cure|100%\s+effective|"
     r"doctors?\s+(?:hate|don'?t\s+want)|big\s+pharma\s+(?:hiding|suppressing))", 0.45),
    # Urgent forward / chain message
    (r"(?:forward\s+to|share\s+with|send\s+to)\s+(?:all|everyone|\d+)", 0.30),
    # "Government will" hoaxes
    (r"(?:government|modi|rbi|rti)\s+(?:will\s+|has\s+)?(?:ban|block|shut\s+down|arrest)\s+"
     r"(?:all|everyone|people\s+who)", 0.35),
    # Free money scams
    (r"(?:free\s+money|free\s+cash|click\s+to\s+claim|limited\s+time|"
     r"act\s+now|don'?t\s+miss|last\s+chance)", 0.40),
]

def _heuristic_misinfo_score(claim: str) -> float:
    """
    Returns an additional false-signal boost (0.0 to 0.6) based on
    linguistic patterns common in misinformation, regardless of specific facts.
    """
    claim_lower = claim.lower()
    total = 0.0
    for pattern, boost in _MISINFO_SIGNAL_PATTERNS:
        if re.search(pattern, claim_lower, re.IGNORECASE):
            total += boost
            logger.debug(f"Misinfo signal pattern matched: boost +{boost}")
    return min(0.60, total)


# ─── Targeted NLI ────────────────────────────────────────────────────────────

def _targeted_nli(claim: str) -> dict:
    """
    Uses domain-specific hypotheses instead of generic labels.
    Returns {'true': float, 'false': float, 'misleading': float}
    """
    nli = _get_nli()
    claim_lower = claim.lower()

    # Choose hypothesis set based on claim domain
    if re.search(r"chip|microchip|gps|tracking|embedded|implant|nano|sensor", claim_lower):
        hypotheses = [
            "This technology claim is scientifically impossible or has no evidence",
            "This is a verified technological fact supported by official sources",
            "This claim about technology is partially true but exaggerated",
        ]
    elif re.search(r"cure|treat|prevent|medicine|vaccine|drug|remedy|health", claim_lower):
        hypotheses = [
            "Medical science contradicts this health claim",
            "Medical evidence supports this health claim",
            "This health claim is an oversimplification of complex medical facts",
        ]
    elif re.search(r"dead|died|killed|death|assassination|passed away", claim_lower):
        hypotheses = [
            "This death claim is false — the person is alive",
            "This death claim is confirmed by official sources",
            "This death claim cannot be verified",
        ]
    elif re.search(r"capital|city|country|state|province", claim_lower):
        hypotheses = [
            "This geographical fact is incorrect",
            "This geographical fact is correct",
            "This geographical claim is partially correct",
        ]
    elif re.search(r"free|government|scheme|yojana|rupee|money|bank|rbi", claim_lower):
        hypotheses = [
            "This government or financial claim is false or a scam",
            "This government policy or scheme is genuine and verified",
            "This financial claim is misleading or exaggerated",
        ]
    elif re.search(r"5g|radiation|electromagnetic|wifi|tower", claim_lower):
        hypotheses = [
            "This technology health claim is scientifically false",
            "This technology claim has scientific support",
            "This technology claim is partially true but overstated",
        ]
    else:
        hypotheses = [
            "This claim is factually incorrect based on available evidence",
            "This claim is factually correct and well-supported",
            "This claim is misleading, exaggerated, or missing important context",
        ]

    try:
        result = nli(claim[:512], candidate_labels=hypotheses)
        scores = dict(zip(result["labels"], result["scores"]))

        true_score = 0.0
        false_score = 0.0
        mislead_score = 0.0

        for label, score in scores.items():
            ll = label.lower()
            if any(w in ll for w in ["incorrect", "false", "impossible", "no evidence",
                                      "scam", "contradicts", "alive", "scientifically false"]):
                false_score = score
            elif any(w in ll for w in ["correct", "genuine", "supported", "verified",
                                        "evidence supports", "confirmed"]):
                true_score = score
            else:
                mislead_score = score

        logger.debug(f"Targeted NLI: true={true_score:.2f} false={false_score:.2f} mislead={mislead_score:.2f}")
        return {"true": true_score, "false": false_score, "misleading": mislead_score}

    except Exception as e:
        logger.warning(f"Targeted NLI failed: {e}")
        return {"true": 0.33, "false": 0.33, "misleading": 0.34}


# ─── Optional external sources ────────────────────────────────────────────────

def _google_fact_check(claim: str) -> Optional[dict]:
    api_key = settings.google_fact_check_api_key
    if not api_key or api_key == "your_google_fact_check_api_key_here":
        return None
    try:
        resp = httpx.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params={"query": claim, "key": api_key, "languageCode": "en"},
            timeout=8,
        )
        data = resp.json()
        claims = data.get("claims", [])
        if not claims:
            return None
        c = claims[0]
        review = c.get("claimReview", [{}])[0]
        rating = review.get("textualRating", "").upper()
        publisher = review.get("publisher", {}).get("name", "Unknown")
        url = review.get("url", "")
        verdict = Verdict.UNVERIFIED
        r = rating.lower()
        if any(w in r for w in ["false", "fake", "wrong", "incorrect", "fabricated", "pants on fire"]):
            verdict = Verdict.FAKE
        elif any(w in r for w in ["true", "correct", "accurate", "verified"]):
            verdict = Verdict.TRUE
        elif any(w in r for w in ["misleading", "partially", "mixed", "mostly false", "half"]):
            verdict = Verdict.MISLEADING
        return {"verdict": verdict, "confidence": 0.94, "source": publisher, "url": url, "rating": rating}
    except Exception as e:
        logger.warning(f"Google FC API error: {e}")
        return None


def _serper_search(claim: str) -> List[str]:
    api_key = settings.serper_api_key
    if not api_key or api_key == "your_serper_api_key_here":
        return []
    try:
        resp = httpx.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": f'fact check "{claim}"', "num": 4},
            timeout=8,
        )
        return [r.get("snippet", "") for r in resp.json().get("organic", []) if r.get("snippet")][:4]
    except Exception as e:
        logger.warning(f"Serper search failed: {e}")
        return []


def _nli_on_context(claim: str, context: str) -> dict:
    """Proper NLI: context as premise, check if it supports/contradicts claim."""
    nli = _get_nli()
    try:
        result = nli(
            context[:512],
            candidate_labels=[
                f"This text confirms that: {claim[:150]}",
                f"This text contradicts the claim that: {claim[:150]}",
                f"This text is unrelated to: {claim[:150]}",
            ]
        )
        scores = dict(zip(result["labels"], result["scores"]))
        confirm = contradict = 0.0
        for label, score in scores.items():
            if "confirms" in label:
                confirm = score
            elif "contradicts" in label:
                contradict = score
        return {"confirm": confirm, "contradict": contradict}
    except Exception as e:
        logger.warning(f"Context NLI failed: {e}")
        return {"confirm": 0.0, "contradict": 0.0}


# ─── Main verify function ─────────────────────────────────────────────────────

def verify_claim(claim: str) -> FactCheckResult:
    """
    Multi-strategy fact verification.

    NOTE: Wikipedia is blocked in this deployment (403 on all requests).
    Pipeline runs in KB + NLI + heuristics mode only.
    External APIs (Google FC, Serper) activate if keys are configured.
    """
    sources = []
    fact_check_urls = []
    explanation_parts = []

    # ── 1. Knowledge base (instant, most accurate) ────────────────
    kb = _knowledge_base_check(claim)
    if kb:
        verdict, confidence, explanation = kb
        return FactCheckResult(
            verdict=verdict,
            confidence=confidence,
            explanation=explanation,
            sources=["Knowledge Base (verified factual database)"],
            fact_check_urls=[],
        )

    # ── 2. Google Fact Check API (authoritative if configured) ─────
    gfc = _google_fact_check(claim)
    if gfc:
        sources.append(f"Fact-checked by: {gfc['source']} (Rating: {gfc['rating']})")
        if gfc["url"]:
            fact_check_urls.append(gfc["url"])
        return FactCheckResult(
            verdict=gfc["verdict"],
            confidence=gfc["confidence"],
            explanation=f"Authoritative fact-check rates this claim as '{gfc['rating']}'.",
            sources=sources,
            fact_check_urls=fact_check_urls,
        )

    # ── 3. Serper web search + context NLI (if configured) ─────────
    search_signal = {"confirm": 0.0, "contradict": 0.0}
    snippets = _serper_search(claim)
    if snippets:
        combined = " ".join(snippets)
        search_signal = _nli_on_context(claim, combined)
        sources.append("Web search (Google)")
        explanation_parts.append(
            f"Web search: {search_signal['contradict']:.0%} contradiction signal."
        )

    # ── 4. Targeted NLI ───────────────────────────────────────────
    nli_scores = _targeted_nli(claim)
    sources.append("NLI Language Model (BART-large-MNLI)")
    explanation_parts.append(
        f"Language model: true={nli_scores['true']:.0%}, "
        f"false={nli_scores['false']:.0%}, misleading={nli_scores['misleading']:.0%}."
    )

    # ── 5. Heuristic misinfo signal ───────────────────────────────
    heuristic_boost = _heuristic_misinfo_score(claim)
    if heuristic_boost > 0:
        explanation_parts.append(
            f"Linguistic patterns associated with misinformation detected (boost: +{heuristic_boost:.0%})."
        )

    # ── 6. Calibrated ensemble ────────────────────────────────────
    has_search = bool(snippets)
    w_search = 0.40 if has_search else 0.0
    w_nli = 1.0 - w_search

    # Base signals
    raw_false = (
        search_signal["contradict"] * w_search +
        nli_scores["false"] * w_nli
    ) + heuristic_boost

    raw_true = (
        search_signal["confirm"] * w_search +
        nli_scores["true"] * w_nli
    )

    raw_misleading = nli_scores["misleading"] * w_nli

    # Cap values
    raw_false    = min(0.98, raw_false)
    raw_true     = min(0.98, raw_true)
    raw_misleading = min(0.98, raw_misleading)

    logger.info(
        f"Ensemble (heuristic_boost={heuristic_boost:.2f}): "
        f"false={raw_false:.2f} true={raw_true:.2f} mislead={raw_misleading:.2f}"
    )

    # ── 7. Decision (NLI-only mode — lower thresholds) ────────────
    if not has_search:
        # Heuristic boost pushes false signal high for conspiracy claims
        if raw_false >= 0.50:
            verdict = Verdict.FAKE
            confidence = round(min(0.92, raw_false + 0.10), 3)
        elif raw_false >= 0.35 and raw_misleading >= 0.20:
            verdict = Verdict.MISLEADING
            confidence = round(min(0.82, raw_misleading + raw_false * 0.3), 3)
        elif raw_misleading >= 0.45:
            verdict = Verdict.MISLEADING
            confidence = round(min(0.82, raw_misleading + 0.08), 3)
        elif raw_true >= 0.50:
            verdict = Verdict.TRUE
            confidence = round(min(0.88, raw_true + 0.08), 3)
        elif raw_false >= 0.30:
            # Low confidence fake — better than UNVERIFIED for suspicious claims
            verdict = Verdict.MISLEADING
            confidence = round(raw_false + 0.10, 3)
        else:
            verdict = Verdict.UNVERIFIED
            confidence = round(max(raw_false, raw_true, raw_misleading), 3)
    else:
        # Ensemble mode with search
        if raw_false >= 0.45:
            verdict = Verdict.FAKE
            confidence = round(min(0.97, raw_false + 0.05), 3)
        elif raw_false >= 0.30 and raw_misleading >= 0.15:
            verdict = Verdict.MISLEADING
            confidence = round(min(0.90, raw_misleading + raw_false * 0.4), 3)
        elif raw_misleading >= 0.45:
            verdict = Verdict.MISLEADING
            confidence = round(min(0.88, raw_misleading + 0.10), 3)
        elif raw_true >= 0.50:
            verdict = Verdict.TRUE
            confidence = round(min(0.95, raw_true + 0.05), 3)
        else:
            verdict = Verdict.UNVERIFIED
            confidence = round(max(raw_false, raw_true, raw_misleading), 3)

    return FactCheckResult(
        verdict=verdict,
        confidence=round(confidence, 3),
        explanation=" ".join(explanation_parts),
        sources=sources,
        fact_check_urls=fact_check_urls,
    )