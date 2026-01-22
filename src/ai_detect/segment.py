"""Person segmentation with multiple backends."""

import logging
import os
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
warnings.filterwarnings("ignore", message=".*trust_remote_code.*")
warnings.filterwarnings("ignore", message=".*new version.*downloaded.*")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@dataclass
class PersonCrop:
    image: Image.Image
    bbox: tuple[int, int, int, int]
    mask: np.ndarray | None = None


class YOLOSegmenter:
    """Fast person detection using YOLO11 - works on CPU/MPS/CUDA."""

    def __init__(self, device: str | None = None):
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self._model = None

    def load(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO("yolo11n.pt")

    def segment_people(
        self,
        image: Image.Image,
        conf: float = 0.5,
        padding: int = 10,
    ) -> list[PersonCrop]:
        """Detect people and return cropped regions."""
        if self._model is None:
            self.load()

        results = self._model.predict(
            image,
            classes=[0],  # person class only
            conf=conf,
            device=self.device,
            verbose=False,
        )

        crops = []
        w, h = image.size

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)

                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue

                cropped = image.crop((x1, y1, x2, y2))
                crops.append(PersonCrop(image=cropped, bbox=(x1, y1, x2, y2)))

        return crops


class Sa2VASegmenter:
    """High-quality segmentation using Sa2VA - CUDA only."""

    MODEL_ID = "ByteDance/Sa2VA-Qwen3-VL-2B"

    def __init__(self, device: str | None = None):
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self._model = None
        self._processor = None

    def load(self) -> None:
        from transformers import AutoModel, AutoProcessor

        if self.device != "cuda":
            raise RuntimeError(
                "Sa2VA segmentation requires CUDA (NVIDIA GPU). "
                "Use --segmenter yolo for CPU/MPS, or use a CUDA-enabled system."
            )

        dtype = torch.bfloat16
        extra_kwargs = {"use_flash_attn": True, "device_map": "cuda"}

        self._model = AutoModel.from_pretrained(
            self.MODEL_ID,
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **extra_kwargs,
        ).eval()

        self._processor = AutoProcessor.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True,
            use_fast=False,
        )

    def segment_people(
        self,
        image: Image.Image,
        padding: int = 10,
    ) -> list[PersonCrop]:
        """Segment all people from an image and return crops with masks."""
        if self._model is None:
            self.load()

        prompt = "<image>Please segment all people in this image."

        input_dict = {
            "image": image,
            "text": prompt,
            "past_text": "",
            "mask_prompts": None,
            "processor": self._processor,
        }

        with torch.no_grad():
            result = self._model.predict_forward(**input_dict)

        masks = result.get("prediction_masks", [])
        if not masks:
            return []

        crops = []
        for mask in masks:
            if isinstance(mask, np.ndarray):
                if mask.ndim == 3:
                    mask = mask[0]

                crop = self._extract_crop(image, mask, padding)
                if crop is not None:
                    crops.append(crop)

        return crops

    def _extract_crop(
        self,
        image: Image.Image,
        mask: np.ndarray,
        padding: int = 10,
    ) -> PersonCrop | None:
        """Extract a cropped region from the mask bounding box."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any() or not cols.any():
            return None

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        w, h = image.size
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        cropped = image.crop((x1, y1, x2, y2))

        return PersonCrop(
            image=cropped,
            bbox=(x1, y1, x2, y2),
            mask=mask,
        )


class PersonSegmenter:
    """Unified segmenter interface with backend selection."""

    def __init__(self, backend: str = "yolo", device: str | None = None):
        self.backend = backend
        self.device = device
        self._segmenter = None

    def load(self) -> None:
        if self.backend == "sa2va":
            self._segmenter = Sa2VASegmenter(self.device)
        else:
            self._segmenter = YOLOSegmenter(self.device)
        self._segmenter.load()

    def segment_people(self, image: Image.Image) -> list[PersonCrop]:
        if self._segmenter is None:
            self.load()
        return self._segmenter.segment_people(image)
