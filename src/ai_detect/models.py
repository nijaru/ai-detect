"""AI image detectors with smart multi-method detection."""

import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import urlretrieve

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

warnings.filterwarnings("ignore", message=".*use_fast.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    is_ai: bool
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)


class ChannelLinear(nn.Linear):
    """Linear layer that works on channel dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return super().forward(x)
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)


def get_cache_dir() -> Path:
    """Get cache directory for model weights."""
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    model_cache = cache_dir / "ai-detect"
    model_cache.mkdir(parents=True, exist_ok=True)
    return model_cache


def download_grip_weights() -> Path | None:
    """Download GRIP-UNINA weights from GitHub."""
    cache_dir = get_cache_dir()
    weights_path = cache_dir / "clipdet_latent10k_plus_weights.pth"

    if weights_path.exists():
        return weights_path

    url = "https://github.com/grip-unina/ClipBased-SyntheticImageDetection/raw/main/weights/clipdet_latent10k_plus/weights.pth"

    try:
        logger.info(f"Downloading GRIP-UNINA weights to {weights_path}")
        urlretrieve(url, weights_path)
        return weights_path
    except Exception as e:
        logger.warning(f"Failed to download GRIP weights: {e}")
        if weights_path.exists():
            weights_path.unlink()
        return None


class ClipDetector(nn.Module):
    """CLIP-based detector using OpenCLIP ViT-L/14 with linear probe.

    Based on GRIP-UNINA's approach from "Raising the Bar of AI-generated
    Image Detection with CLIP" (CVPRW 2024).
    """

    def __init__(self, device: str | None = None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._clip = None
        self._head = None
        self._transform = None
        self._weights_loaded = False

    def load(self) -> None:
        import open_clip
        from huggingface_hub import hf_hub_download

        # Load CLIP backbone
        clip_model = open_clip.create_model(
            "ViT-L-14",
            pretrained=hf_hub_download(
                "laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K",
                "open_clip_pytorch_model.bin",
            ),
        )
        num_features = clip_model.visual.proj.shape[0]
        clip_model.visual.proj = None
        self._clip = clip_model.to(self.device).eval()

        # Create linear head
        self._head = ChannelLinear(num_features, 1)

        # Try to load GRIP-UNINA weights
        weights_path = download_grip_weights()
        if weights_path:
            try:
                state = torch.load(weights_path, map_location="cpu", weights_only=True)
                if "_fc.weight" in state:
                    self._head.weight.data = state["_fc.weight"]
                    self._head.bias.data = state["_fc.bias"]
                    self._weights_loaded = True
                    logger.info("Loaded GRIP-UNINA detection weights")
                elif "main.0.weight" in state:
                    self._head.weight.data = state["main.0.weight"]
                    self._head.bias.data = state["main.0.bias"]
                    self._weights_loaded = True
                    logger.info("Loaded GRIP-UNINA detection weights")
            except Exception as e:
                logger.warning(f"Failed to load weights: {e}")

        if not self._weights_loaded:
            logger.warning("Using random initialization for detection head")
            nn.init.normal_(self._head.weight.data, 0.0, 0.02)
            nn.init.zeros_(self._head.bias.data)

        self._head = self._head.to(self.device)

        self._transform = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self._clip.encode_image(x, normalize=True)
        return self._head(features)

    def detect(self, image: Image.Image) -> DetectionResult:
        if self._clip is None:
            self.load()

        x = self._transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.forward(x)

        llr = logit.squeeze().item()
        prob = torch.sigmoid(torch.tensor(llr)).item()

        is_ai = llr > 0
        confidence = prob if is_ai else (1 - prob)

        return DetectionResult(
            is_ai=is_ai,
            confidence=confidence,
            scores={"ai": prob, "real": 1 - prob, "llr": llr},
        )

    def detect_batch(self, images: list[Image.Image]) -> list[DetectionResult]:
        """Batch detection for multiple images."""
        if self._clip is None:
            self.load()

        if not images:
            return []

        # Transform all images
        batch = torch.stack([self._transform(img) for img in images]).to(self.device)

        with torch.no_grad():
            logits = self.forward(batch)

        results = []
        for i in range(len(images)):
            llr = logits[i].squeeze().item()
            prob = torch.sigmoid(torch.tensor(llr)).item()
            is_ai = llr > 0
            confidence = prob if is_ai else (1 - prob)
            results.append(
                DetectionResult(
                    is_ai=is_ai,
                    confidence=confidence,
                    scores={"ai": prob, "real": 1 - prob, "llr": llr},
                )
            )

        return results


class SigLIPDetector:
    """SigLIP-based detector using Ateeqq model (fallback/fast)."""

    MODEL_ID = "Ateeqq/ai-vs-human-image-detector"

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def load(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self._processor = AutoImageProcessor.from_pretrained(
            self.MODEL_ID, use_fast=True
        )
        self._model = AutoModelForImageClassification.from_pretrained(self.MODEL_ID)
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

        scores_lower = {k.lower(): v for k, v in scores.items()}
        ai_score = scores_lower.get("ai", 0.0)
        human_score = scores_lower.get("hum", scores_lower.get("human", 0.0))

        is_ai = ai_score > human_score
        confidence = ai_score if is_ai else human_score

        return DetectionResult(is_ai=is_ai, confidence=confidence, scores=scores)

    def detect_batch(self, images: list[Image.Image]) -> list[DetectionResult]:
        """Batch detection for multiple images."""
        if self._model is None:
            self.load()

        if not images:
            return []

        inputs = self._processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        labels = self._model.config.id2label
        results = []

        for i in range(len(images)):
            scores = {labels[j]: probs[i][j].item() for j in range(len(labels))}
            scores_lower = {k.lower(): v for k, v in scores.items()}
            ai_score = scores_lower.get("ai", 0.0)
            human_score = scores_lower.get("hum", scores_lower.get("human", 0.0))
            is_ai = ai_score > human_score
            confidence = ai_score if is_ai else human_score
            results.append(
                DetectionResult(is_ai=is_ai, confidence=confidence, scores=scores)
            )

        return results


class Detector:
    """Main detector with smart multi-method detection.

    Modes:
        - smart (default): Full image + auto person detection + strategic patches
        - fast: Full image only
        - thorough: All methods including frequency analysis
    """

    PATCH_SIZE = 224
    LARGE_IMAGE_THRESHOLD = 1024
    MAX_PEOPLE = 3
    BATCH_SIZE = 8

    def __init__(
        self,
        device: str | None = None,
        backend: str = "clip",
        mode: str = "smart",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = backend
        self.mode = mode
        self._detector = None
        self._yolo = None

    def load(self) -> None:
        if self.backend == "clip":
            self._detector = ClipDetector(self.device)
        else:
            self._detector = SigLIPDetector(self.device)
        self._detector.load()

    def _get_yolo(self):
        """Lazy load YOLO for person detection."""
        if self._yolo is None:
            from .segment import YOLOSegmenter

            self._yolo = YOLOSegmenter(self.device)
            self._yolo.load()
        return self._yolo

    def detect(self, image: Image.Image) -> DetectionResult:
        """Route to appropriate detection method based on mode."""
        if self._detector is None:
            self.load()

        if self.mode == "fast":
            return self._detect_full(image)
        elif self.mode == "thorough":
            return self._detect_thorough(image)
        else:
            return self._detect_smart(image)

    def _detect_full(self, image: Image.Image) -> DetectionResult:
        """Full image detection only."""
        return self._detector.detect(image)

    def _detect_smart(self, image: Image.Image) -> DetectionResult:
        """Smart detection: full image + people + strategic patches with batching."""
        # Collect all images to process
        images_to_process = [("full_image", image)]

        # Person detection (YOLO is fast, ~50ms)
        yolo = self._get_yolo()
        people = yolo.segment_people(image)

        if people:
            people_sorted = sorted(
                people,
                key=lambda p: (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]),
                reverse=True,
            )
            for i, person in enumerate(people_sorted[: self.MAX_PEOPLE]):
                images_to_process.append((f"person_{i + 1}", person.image))

        # Strategic patches for large images
        w, h = image.size
        if w > self.LARGE_IMAGE_THRESHOLD or h > self.LARGE_IMAGE_THRESHOLD:
            patches = self._get_strategic_patches(image)
            for i, patch in enumerate(patches):
                images_to_process.append((f"patch_{i + 1}", patch))

        # Batch process all images
        method_scores = {}
        images = [img for _, img in images_to_process]
        labels = [label for label, _ in images_to_process]

        # Process in batches
        for i in range(0, len(images), self.BATCH_SIZE):
            batch_images = images[i : i + self.BATCH_SIZE]
            batch_labels = labels[i : i + self.BATCH_SIZE]
            batch_results = self._detector.detect_batch(batch_images)

            for label, result in zip(batch_labels, batch_results):
                method_scores[label] = result.scores.get("ai", 0)

        # Aggregate: take max AI score
        max_ai = max(method_scores.values())
        is_ai = max_ai > 0.5
        confidence = max_ai if is_ai else (1 - max_ai)

        return DetectionResult(
            is_ai=is_ai,
            confidence=confidence,
            scores={
                "ai": max_ai,
                "real": 1 - max_ai,
                "methods": method_scores,
            },
        )

    def _detect_thorough(self, image: Image.Image) -> DetectionResult:
        """Thorough detection: all methods including frequency analysis."""
        # Start with smart detection
        result = self._detect_smart(image)
        method_scores = dict(result.scores.get("methods", {}))

        # Add dense patches if image is large enough
        w, h = image.size
        if w >= self.PATCH_SIZE * 2 and h >= self.PATCH_SIZE * 2:
            dense_patches = self._get_dense_patches(image, stride=self.PATCH_SIZE)

            # Filter out patches we already have
            new_patches = []
            new_labels = []
            for i, patch in enumerate(dense_patches):
                label = f"dense_patch_{i + 1}"
                if label not in method_scores:
                    new_patches.append(patch)
                    new_labels.append(label)

            # Batch process new patches
            for i in range(0, len(new_patches), self.BATCH_SIZE):
                batch_images = new_patches[i : i + self.BATCH_SIZE]
                batch_labels = new_labels[i : i + self.BATCH_SIZE]
                batch_results = self._detector.detect_batch(batch_images)

                for label, res in zip(batch_labels, batch_results):
                    method_scores[label] = res.scores.get("ai", 0)

        # Add frequency analysis
        from .frequency import analyze_frequency

        freq_features = analyze_frequency(image)
        method_scores["frequency"] = freq_features["freq_score"]

        # Re-aggregate (excluding frequency from max)
        neural_scores = [v for k, v in method_scores.items() if k != "frequency"]
        max_ai = max(neural_scores) if neural_scores else 0.5
        is_ai = max_ai > 0.5
        confidence = max_ai if is_ai else (1 - max_ai)

        return DetectionResult(
            is_ai=is_ai,
            confidence=confidence,
            scores={
                "ai": max_ai,
                "real": 1 - max_ai,
                "freq_score": freq_features["freq_score"],
                "methods": method_scores,
            },
        )

    def _get_strategic_patches(self, image: Image.Image) -> list[Image.Image]:
        """Get 5 strategic patches: 4 corners + center."""
        w, h = image.size
        ps = self.PATCH_SIZE

        if w < ps or h < ps:
            return []

        positions = [
            (0, 0),
            (w - ps, 0),
            (0, h - ps),
            (w - ps, h - ps),
            ((w - ps) // 2, (h - ps) // 2),
        ]

        return [image.crop((x, y, x + ps, y + ps)) for x, y in positions]

    def _get_dense_patches(
        self, image: Image.Image, stride: int = 112
    ) -> list[Image.Image]:
        """Get dense patches for thorough mode."""
        w, h = image.size
        ps = self.PATCH_SIZE

        patches = []
        for y in range(0, h - ps + 1, stride):
            for x in range(0, w - ps + 1, stride):
                patches.append(image.crop((x, y, x + ps, y + ps)))

        return patches
