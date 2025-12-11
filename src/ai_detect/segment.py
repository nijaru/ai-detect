"""Person segmentation using Sa2VA-Qwen3-VL."""

import logging
import os
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

# Suppress verbose HuggingFace warnings
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"  # Keep progress bars
warnings.filterwarnings("ignore", message=".*trust_remote_code.*")
warnings.filterwarnings("ignore", message=".*new version.*downloaded.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

SA2VA_MODEL_ID = "ByteDance/Sa2VA-Qwen3-VL-1B"


@dataclass
class PersonCrop:
    image: Image.Image
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: np.ndarray


class PersonSegmenter:
    """Segment people from images using Sa2VA-Qwen3-VL."""

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
        self._processor = None

    def load(self) -> None:
        from transformers import AutoModel, AutoProcessor

        # Sa2VA custom code requires CUDA - cannot run on MPS/CPU
        if self.device != "cuda":
            raise RuntimeError(
                "Sa2VA segmentation requires CUDA (NVIDIA GPU). "
                "Run without --subjects/-S on this machine, or use a CUDA-enabled system."
            )

        dtype = torch.bfloat16
        extra_kwargs = {"use_flash_attn": True, "device_map": "cuda"}

        self._model = AutoModel.from_pretrained(
            SA2VA_MODEL_ID,
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **extra_kwargs,
        ).eval()

        self._processor = AutoProcessor.from_pretrained(
            SA2VA_MODEL_ID,
            trust_remote_code=True,
            use_fast=False,
        )

    def segment_people(self, image: Image.Image) -> list[PersonCrop]:
        """Segment all people from an image and return crops."""
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
                # Handle shape: could be (1, h, w) or (h, w)
                if mask.ndim == 3:
                    mask = mask[0]

                crop = self._extract_crop(image, mask)
                if crop is not None:
                    crops.append(crop)

        return crops

    def _extract_crop(
        self, image: Image.Image, mask: np.ndarray, padding: int = 10
    ) -> PersonCrop | None:
        """Extract a cropped region from the mask bounding box."""
        # Find bounding box of mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any() or not cols.any():
            return None

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        # Add padding
        w, h = image.size
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        # Crop image
        cropped = image.crop((x1, y1, x2, y2))

        return PersonCrop(
            image=cropped,
            bbox=(x1, y1, x2, y2),
            mask=mask,
        )
