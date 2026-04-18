from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import io

# -----------------------------
# LOAD MODEL (ONCE)
# -----------------------------
try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    print("BLIP caption model loaded")
except Exception as e:
    processor = None
    model = None
    print("BLIP load failed:", e)


# -----------------------------
# GENERATE CAPTION
# -----------------------------
def generate_caption(image_bytes: bytes):
    if processor is None or model is None:
        return ""

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(**inputs)

        caption = processor.decode(output[0], skip_special_tokens=True)

        print(f"Generated caption: {caption}")

        return caption

    except Exception as e:
        print("Captioning error:", e)
        return ""