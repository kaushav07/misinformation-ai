"""
RoBERTa-based AI-generated text detector.

Model: roberta-base-openai-detector
- Trained to distinguish human-written vs GPT-generated text
- Input: up to 512 tokens
- Output: REAL (human) | FAKE (AI-generated) with confidence

For hackathon: uses HuggingFace transformers pipeline.
For production: fine-tune on domain-specific Indian news data.
"""
from typing import Optional


class AITextDetector:
    """
    Wrapper around roberta-base-openai-detector for AI content detection.
    Includes heuristic fallback when model is unavailable.
    """

    def __init__(self, model_name: str = "roberta-base-openai-detector"):
        self.model_name = model_name
        self._pipeline = None
        self._loaded = False

    def load(self):
        """Lazy-load the model."""
        if self._loaded:
            return
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=-1,  # CPU
                truncation=True,
                max_length=512,
            )
            self._loaded = True
        except Exception as e:
            from loguru import logger
            logger.warning(f"Could not load AI detector model: {e}. Heuristic fallback will be used.")
            self._loaded = True  # mark as loaded to prevent retry spam

    def predict(self, text: str) -> dict:
        """
        Returns:
            {"ai_probability": float, "human_probability": float, "label": str}
        """
        self.load()

        if self._pipeline is not None:
            return self._predict_model(text)
        return self._predict_heuristic(text)

    def _predict_model(self, text: str) -> dict:
        result = self._pipeline(text[:1024])
        label = result[0]["label"].upper()  # "FAKE" or "REAL"
        score = result[0]["score"]

        if "FAKE" in label:
            ai_prob = score
        else:
            ai_prob = 1.0 - score

        return {
            "ai_probability": round(ai_prob, 4),
            "human_probability": round(1.0 - ai_prob, 4),
            "label": "AI_GENERATED" if ai_prob >= 0.5 else "HUMAN_WRITTEN",
            "confidence": round(max(ai_prob, 1.0 - ai_prob), 4),
        }

    def _predict_heuristic(self, text: str) -> dict:
        """Lightweight heuristic AI detection."""
        import re
        score = 0.0
        lower = text.lower()

        # Uniform sentence lengths
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 10]
        if len(sentences) >= 3:
            lengths = [len(s.split()) for s in sentences]
            mean = sum(lengths) / len(lengths)
            variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
            if variance < 8:
                score += 0.25

        # AI phrases
        ai_phrases = [
            "it is important to note", "in conclusion", "furthermore",
            "it should be noted", "in summary", "to summarize",
            "as an ai", "as a language model", "in this context",
            "it is worth mentioning", "it is crucial to", "i cannot",
            "i don't have access", "based on the information provided",
        ]
        matches = sum(1 for p in ai_phrases if p in lower)
        score += min(0.5, matches * 0.12)

        # Very long words ratio (AI tends to be verbose)
        words = text.split()
        long_words = sum(1 for w in words if len(w) > 10)
        if words:
            long_ratio = long_words / len(words)
            if long_ratio > 0.15:
                score += 0.15

        ai_prob = min(1.0, score)
        return {
            "ai_probability": round(ai_prob, 4),
            "human_probability": round(1.0 - ai_prob, 4),
            "label": "AI_GENERATED" if ai_prob >= 0.5 else "HUMAN_WRITTEN",
            "confidence": round(max(ai_prob, 1.0 - ai_prob), 4),
        }


_detector_instance: Optional[AITextDetector] = None


def get_detector() -> AITextDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = AITextDetector()
    return _detector_instance
