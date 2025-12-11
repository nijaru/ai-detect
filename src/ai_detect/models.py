"""AI image detector using Ateeqq/ai-vs-human-image-detector."""

import logging
import warnings
from dataclasses import dataclass

import torch
from PIL import Image

# Suppress verbose HuggingFace warnings
warnings.filterwarnings("ignore", message=".*use_fast.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

MODEL_ID = "Ateeqq/ai-vs-human-image-detector"


@dataclass
class DetectionResult:
    is_ai: bool
    confidence: float
    scores: dict[str, float]


class Detector:
    """AI image detector using SigLIP model trained on FLUX/MJ6.1/SD3.5/GPT-4o."""

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def load(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self._processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
        self._model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
        self._model.to(self.device)
        self._model.eval()

    def detect(self, image: Image.Image) -> DetectionResult:
        if self._model is None:
            self.load()

        inputs = self._processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        labels = self._model.config.id2label
        scores = {labels[i]: probs[0][i].item() for i in range(len(labels))}

        # Normalize label lookup (model uses "ai" and "hum")
        scores_lower = {k.lower(): v for k, v in scores.items()}
        ai_score = scores_lower.get("ai", 0.0)
        human_score = scores_lower.get("hum", scores_lower.get("human", 0.0))

        if ai_score == 0.0 and human_score == 0.0:
            logger.warning(f"Unknown model labels: {list(scores.keys())}")

        is_ai = ai_score > human_score
        confidence = ai_score if is_ai else human_score

        return DetectionResult(is_ai=is_ai, confidence=confidence, scores=scores)
