"""
AI-generated text detection inference script.

Usage:
    python ml/ai_detect/inference.py "Some text to check"
    python ml/ai_detect/inference.py --file path/to/article.txt
    python ml/ai_detect/inference.py --batch texts.json
"""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.ai_detect.model import get_detector


def analyse_text(text: str) -> dict:
    detector = get_detector()
    return detector.predict(text)


def analyse_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return analyse_text(text)


def analyse_batch(json_path: str) -> list:
    """Expects a JSON file with {"texts": ["text1", "text2", ...]}"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = data.get("texts", [])
    return [analyse_text(t) for t in texts]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI text detection")
    parser.add_argument("text", nargs="?", help="Text to analyse")
    parser.add_argument("--file", help="Path to text file")
    parser.add_argument("--batch", help="Path to JSON batch file")
    args = parser.parse_args()

    if args.batch:
        results = analyse_batch(args.batch)
        for i, r in enumerate(results):
            print(f"\n[{i+1}] {r}")
    elif args.file:
        result = analyse_file(args.file)
        print(json.dumps(result, indent=2))
    elif args.text:
        result = analyse_text(args.text)
        print(f"\n{'='*40}")
        for k, v in result.items():
            print(f"  {k:25s}: {v}")
        print(f"{'='*40}\n")
    else:
        parser.print_help()
