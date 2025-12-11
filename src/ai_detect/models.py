"""Detector model implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch
from PIL import Image


@dataclass
class DetectionResult:
    """Result from a single detector."""

    model_name: str
    is_ai: bool
    confidence: float
    raw_scores: dict[str, float] | None = None


class BaseDetector(ABC):
    """Base class for AI image detectors."""

    name: ClassVar[str]
    description: ClassVar[str]

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    @abstractmethod
    def load(self) -> None:
        """Load model weights."""

    @abstractmethod
    def detect(self, image: Image.Image) -> DetectionResult:
        """Run detection on an image."""

    def is_loaded(self) -> bool:
        return self._model is not None


class AteeqDetector(BaseDetector):
    """Ateeqq/ai-vs-human-image-detector - SigLIP-based, trained on FLUX/MJ6.1/SD3.5/GPT-4o."""

    name = "Ateeqq/ai-vs-human"
    description = "Latest model trained on FLUX 1.1, Midjourney v6.1, SD 3.5, GPT-4o"

    def load(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        model_id = "Ateeqq/ai-vs-human-image-detector"
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModelForImageClassification.from_pretrained(model_id)
        self._model.to(self.device)
        self._model.eval()

    def detect(self, image: Image.Image) -> DetectionResult:
        if not self.is_loaded():
            self.load()

        inputs = self._processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        labels = self._model.config.id2label
        scores = {labels[i]: probs[0][i].item() for i in range(len(labels))}

        ai_score = scores.get("AI", scores.get("ai", scores.get("artificial", 0.0)))
        human_score = scores.get("Human", scores.get("human", scores.get("real", 0.0)))

        if ai_score == 0.0 and human_score == 0.0:
            ai_score = max(scores.values())

        is_ai = ai_score > human_score
        confidence = ai_score if is_ai else human_score

        return DetectionResult(
            model_name=self.name,
            is_ai=is_ai,
            confidence=confidence,
            raw_scores=scores,
        )


class GripUninaDetector(BaseDetector):
    """GRIP-UNINA ClipBased-SyntheticImageDetection - CLIP ViT-L/14 based."""

    name = "GRIP-UNINA/CLIP"
    description = "CVPR 2024 SOTA, excellent OOD generalization"

    def load(self) -> None:
        import open_clip

        self._model, _, self._processor = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self._model.to(self.device)
        self._model.eval()

        self._fc = torch.nn.Linear(768, 1)
        weights_path = Path.home() / ".cache" / "ai-detect" / "grip_unina_fc.pth"

        if weights_path.exists():
            self._fc.load_state_dict(torch.load(weights_path, map_location=self.device))
        self._fc.to(self.device)

    def detect(self, image: Image.Image) -> DetectionResult:
        if not self.is_loaded():
            self.load()

        img_tensor = self._processor(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self._model.encode_image(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            logit = self._fc(features.float())
            prob = torch.sigmoid(logit).item()

        return DetectionResult(
            model_name=self.name,
            is_ai=prob > 0.5,
            confidence=prob if prob > 0.5 else 1 - prob,
            raw_scores={"ai_probability": prob},
        )


class OrganikaDetector(BaseDetector):
    """Organika/sdxl-detector - Swin Transformer, SDXL specialized."""

    name = "Organika/sdxl"
    description = "SDXL/modern diffusion specialized"

    def load(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        model_id = "Organika/sdxl-detector"
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModelForImageClassification.from_pretrained(model_id)
        self._model.to(self.device)
        self._model.eval()

    def detect(self, image: Image.Image) -> DetectionResult:
        if not self.is_loaded():
            self.load()

        inputs = self._processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        labels = self._model.config.id2label
        scores = {labels[i]: probs[0][i].item() for i in range(len(labels))}

        ai_score = scores.get("artificial", scores.get("AI", scores.get("ai", 0.0)))
        real_score = scores.get("real", scores.get("Human", scores.get("human", 0.0)))

        if ai_score == 0.0 and real_score == 0.0:
            ai_score = max(scores.values())

        is_ai = ai_score > real_score
        confidence = ai_score if is_ai else real_score

        return DetectionResult(
            model_name=self.name,
            is_ai=is_ai,
            confidence=confidence,
            raw_scores=scores,
        )


class DeepFakeDetector(BaseDetector):
    """prithivMLmods/Deep-Fake-Detector-v2-Model - ViT-based deepfake detection."""

    name = "DeepFake-v2"
    description = "ViT-based deepfake focus"

    def load(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        model_id = "prithivMLmods/Deep-Fake-Detector-v2-Model"
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModelForImageClassification.from_pretrained(model_id)
        self._model.to(self.device)
        self._model.eval()

    def detect(self, image: Image.Image) -> DetectionResult:
        if not self.is_loaded():
            self.load()

        inputs = self._processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        labels = self._model.config.id2label
        scores = {labels[i]: probs[0][i].item() for i in range(len(labels))}

        fake_keys = ["fake", "Fake", "AI", "ai", "artificial", "deepfake"]
        real_keys = ["real", "Real", "Human", "human", "authentic"]

        fake_score = next((scores[k] for k in fake_keys if k in scores), 0.0)
        real_score = next((scores[k] for k in real_keys if k in scores), 0.0)

        if fake_score == 0.0 and real_score == 0.0:
            fake_score = max(scores.values())

        is_ai = fake_score > real_score
        confidence = fake_score if is_ai else real_score

        return DetectionResult(
            model_name=self.name,
            is_ai=is_ai,
            confidence=confidence,
            raw_scores=scores,
        )


class UmmMaybeDetector(BaseDetector):
    """umm-maybe/AI-image-detector - artistic AI images baseline."""

    name = "umm-maybe"
    description = "Original artistic AI detector baseline"

    def load(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        model_id = "umm-maybe/AI-image-detector"
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModelForImageClassification.from_pretrained(model_id)
        self._model.to(self.device)
        self._model.eval()

    def detect(self, image: Image.Image) -> DetectionResult:
        if not self.is_loaded():
            self.load()

        inputs = self._processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        labels = self._model.config.id2label
        scores = {labels[i]: probs[0][i].item() for i in range(len(labels))}

        ai_keys = ["artificial", "AI", "ai", "fake"]
        human_keys = ["human", "Human", "real", "Real"]

        ai_score = next((scores[k] for k in ai_keys if k in scores), 0.0)
        human_score = next((scores[k] for k in human_keys if k in scores), 0.0)

        if ai_score == 0.0 and human_score == 0.0:
            ai_score = max(scores.values())

        is_ai = ai_score > human_score
        confidence = ai_score if is_ai else human_score

        return DetectionResult(
            model_name=self.name,
            is_ai=is_ai,
            confidence=confidence,
            raw_scores=scores,
        )


DETECTORS: dict[str, type[BaseDetector]] = {
    "ateeq": AteeqDetector,
    "grip": GripUninaDetector,
    "organika": OrganikaDetector,
    "deepfake": DeepFakeDetector,
    "umm-maybe": UmmMaybeDetector,
}

FAST_DETECTORS = ["ateeq", "grip"]
ALL_DETECTORS = list(DETECTORS.keys())


def get_detector(name: str, device: str | None = None) -> BaseDetector:
    """Get a detector instance by name."""
    if name not in DETECTORS:
        raise ValueError(
            f"Unknown detector: {name}. Available: {list(DETECTORS.keys())}"
        )
    return DETECTORS[name](device=device)


def get_detectors(mode: str = "all", device: str | None = None) -> list[BaseDetector]:
    """Get detector instances based on mode."""
    names = FAST_DETECTORS if mode == "fast" else ALL_DETECTORS
    return [get_detector(name, device) for name in names]
