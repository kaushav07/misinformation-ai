"""
/api/v1/segmentation — Special Problem Statement: Duality AI Offroad Segmentation.

This router integrates the segmentation model results with Qdrant:
- Store class embeddings for semantic analysis
- Query similar misclassified classes
- Serve IoU scores and failure case analysis via API
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import SegmentationResult

router = APIRouter()

# Segmentation class map (from problem statement)
SEG_CLASSES = {
    100:  "Trees",
    200:  "Lush Bushes",
    300:  "Dry Grass",
    500:  "Dry Bushes",
    550:  "Ground Clutter",
    600:  "Flowers",
    700:  "Logs",
    800:  "Rocks",
    7100: "Landscape",
    10000:"Sky",
}


@router.get("/segmentation/classes", summary="List segmentation classes (Duality AI)")
def list_classes():
    """Returns all 10 semantic segmentation classes from the Duality AI challenge."""
    return {
        "challenge": "Duality AI Offroad Autonomy Segmentation",
        "environment": "Desert (synthetic — FalconEditor)",
        "classes": [
            {"class_id": k, "class_name": v}
            for k, v in SEG_CLASSES.items()
        ],
        "total_classes": len(SEG_CLASSES),
    }


@router.post("/segmentation/analyze", response_model=SegmentationResult, summary="Analyse segmentation predictions")
async def analyze_segmentation(
    prediction_file: UploadFile = File(..., description="Prediction mask image (PNG)"),
    ground_truth_file: UploadFile = File(..., description="Ground truth mask image (PNG)"),
):
    """
    Computes per-class IoU between prediction and ground truth masks.
    Also indexes class embeddings into Qdrant for failure analysis.

    Returns:
    - Overall mean IoU
    - Per-class IoU scores
    - Failure cases (classes with IoU < 0.3)
    """
    try:
        import numpy as np
        from PIL import Image
        import io

        pred_bytes = await prediction_file.read()
        gt_bytes = await ground_truth_file.read()

        pred_img = np.array(Image.open(io.BytesIO(pred_bytes)))
        gt_img = np.array(Image.open(io.BytesIO(gt_bytes)))

        if pred_img.shape != gt_img.shape:
            raise HTTPException(status_code=422, detail="Prediction and ground truth must be same dimensions.")

        class_ious = {}
        failure_cases = []

        for class_id, class_name in SEG_CLASSES.items():
            pred_mask = (pred_img == class_id)
            gt_mask = (gt_img == class_id)

            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()

            if union == 0:
                continue  # class not present in either

            iou = float(intersection) / float(union)
            class_ious[class_name] = round(iou, 4)

            if iou < 0.30:
                failure_cases.append(
                    f"{class_name} (ID={class_id}): IoU={iou:.3f} — likely misclassified. "
                    f"Check for occlusion or class imbalance."
                )

        mean_iou = round(float(np.mean(list(class_ious.values()))), 4) if class_ious else 0.0

        # Index class embeddings in Qdrant for semantic failure analysis
        try:
            from app.services.qdrant_service import _get_client, _embed
            client = _get_client()
            from qdrant_client.models import VectorParams, Distance, PointStruct
            import uuid

            # Create seg_classes collection if needed
            existing = {c.name for c in client.get_collections().collections}
            if "seg_classes" not in existing:
                client.create_collection(
                    collection_name="seg_classes",
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )

            for class_name, iou_score in class_ious.items():
                vector = _embed(f"desert environment {class_name} segmentation class")
                client.upsert(
                    collection_name="seg_classes",
                    points=[PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "class_name": class_name,
                            "iou_score": iou_score,
                            "is_failure": iou_score < 0.30,
                        }
                    )]
                )
        except Exception:
            pass  # Don't fail if Qdrant not available

        return SegmentationResult(
            iou_score=mean_iou,
            class_scores=class_ious,
            failure_cases=failure_cases,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation analysis failed: {str(e)}")


@router.get("/segmentation/similar-classes", summary="Find semantically similar misclassified classes")
def find_similar_classes(class_name: str, top_k: int = 3):
    """
    **UNIQUE QDRANT FEATURE**: Uses semantic similarity to find classes
    that are frequently confused with each other in the segmentation model.

    E.g., 'Dry Grass' and 'Dry Bushes' are semantically similar and likely
    to be confused — this helps diagnose model weaknesses.
    """
    try:
        from app.services.qdrant_service import _get_client, _embed
        client = _get_client()
        vector = _embed(f"desert environment {class_name} segmentation class")

        results = client.query_points(
            collection_name="seg_classes",
            query=vector,
            limit=top_k + 1,
            with_payload=True,
        )

        similar = [
            {
                "class_name": p.payload.get("class_name"),
                "iou_score": p.payload.get("iou_score"),
                "similarity": round(p.score, 3),
                "is_failure": p.payload.get("is_failure", False),
            }
            for p in results.points
            if p.payload.get("class_name") != class_name
        ][:top_k]

        return {
            "query_class": class_name,
            "similar_classes": similar,
            "insight": f"Classes similar to '{class_name}' may cause misclassification. Consider augmenting training data for these pairs.",
        }
    except Exception as e:
        return {"error": str(e), "query_class": class_name, "similar_classes": []}


@router.get("/segmentation/report-template", summary="Get IoU report template")
def get_report_template():
    """Returns a structured template for the Duality AI segmentation report."""
    return {
        "report_format": {
            "max_pages": 8,
            "required_sections": [
                {"page": 1, "section": "Title", "content": "Team name, project name, brief tagline"},
                {"page": 2, "section": "Methodology", "content": "Training steps, model choice, fine-tuning strategy"},
                {"pages": "3-4", "section": "Results & Performance", "content": "IoU score, confusion matrix, accuracy comparisons"},
                {"pages": "5-6", "section": "Challenges & Solutions", "content": "Key obstacles and how resolved"},
                {"page": 7, "section": "Conclusion & Future Work", "content": "Final thoughts, potential improvements"},
            ],
            "evaluation_criteria": {
                "IoU_score": {"points": 80, "description": "Pixel-level classification accuracy"},
                "report_clarity": {"points": 20, "description": "Structured findings and detailed reporting"},
            },
        },
        "tips": [
            "Use augmentation (random flip, color jitter) to improve recall on minority classes like 'Logs'.",
            "Class imbalance is most severe for Flowers (600) and Ground Clutter (550) — oversample or use weighted loss.",
            "Inference speed target: < 50ms per image.",
            "Store CLIP embeddings of class samples in Qdrant for semantic failure diagnosis (Qdrant prize track).",
        ],
    }
