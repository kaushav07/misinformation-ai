"""
Deepfake inference pipeline.

Usage:
    python ml/deepfake/inference.py path/to/image.jpg
    python ml/deepfake/inference.py path/to/video.mp4

Returns:
    deepfake_probability: float [0, 1]
    verdict: LIKELY DEEPFAKE | UNCERTAIN | LIKELY AUTHENTIC
"""
import sys
import os
import io
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.deepfake.model import get_model


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def predict_image(image_path: str) -> dict:
    """Predict deepfake probability for a single image file."""
    model = get_model()
    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0)

    probs = model.predict_proba(tensor)
    fake_prob = probs[0][1].item()

    return _format_result(fake_prob)


def predict_image_bytes(image_bytes: bytes) -> dict:
    """Predict deepfake probability from raw image bytes."""
    model = get_model()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0)

    probs = model.predict_proba(tensor)
    fake_prob = probs[0][1].item()

    return _format_result(fake_prob)


def predict_video(video_path: str, sample_frames: int = 10) -> dict:
    """
    Sample frames from a video and average deepfake probabilities.
    Returns the max probability across frames (conservative approach).
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // sample_frames)

    scores = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            _, buf = cv2.imencode(".jpg", frame)
            result = predict_image_bytes(buf.tobytes())
            scores.append(result["deepfake_probability"])
        idx += 1

    cap.release()

    if not scores:
        return _format_result(0.0)

    # Use max (most suspicious frame) rather than mean
    max_prob = float(np.max(scores))
    mean_prob = float(np.mean(scores))
    return {
        **_format_result(max_prob),
        "mean_probability": round(mean_prob, 3),
        "frames_analysed": len(scores),
    }


def _format_result(fake_prob: float) -> dict:
    fake_prob = round(fake_prob, 4)
    if fake_prob >= 0.65:
        verdict = "LIKELY DEEPFAKE"
    elif fake_prob <= 0.35:
        verdict = "LIKELY AUTHENTIC"
    else:
        verdict = "UNCERTAIN"

    return {
        "deepfake_probability": fake_prob,
        "authentic_probability": round(1.0 - fake_prob, 4),
        "verdict": verdict,
        "confidence": round(max(fake_prob, 1.0 - fake_prob), 4),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_or_video_path>")
        sys.exit(1)

    path = sys.argv[1]
    ext = os.path.splitext(path)[1].lower()

    if ext in (".mp4", ".avi", ".mov", ".webm"):
        result = predict_video(path)
    else:
        result = predict_image(path)

    print(f"\n{'='*40}")
    print(f"File   : {path}")
    for k, v in result.items():
        print(f"{k:30s}: {v}")
    print(f"{'='*40}\n")
