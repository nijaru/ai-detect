# CPU-Compatible Person/Face Segmentation Models

Research conducted: 2025-12-27

## Context

Current implementation uses `ByteDance/Sa2VA-Qwen3-VL-2B` which requires CUDA. Need alternatives for CPU and Apple Silicon MPS.

## Recommendation Summary

| Option                            | Speed (CPU) | Quality   | Complexity | Best For                                    |
| --------------------------------- | ----------- | --------- | ---------- | ------------------------------------------- |
| **Ultralytics YOLO + Crop**       | ~50-200ms   | High      | Low        | Simple person detection, bounding box crops |
| **MediaPipe Selfie Segmentation** | ~30-100ms   | Medium    | Low        | Single person, full-body mask               |
| **YOLOv8-seg**                    | ~100-300ms  | High      | Low        | Instance segmentation with masks            |
| **FastSAM**                       | ~200-500ms  | High      | Medium     | Flexible prompting, quality masks           |
| **MobileSAM**                     | ~300-600ms  | Very High | Medium     | Best quality masks, slower                  |

## Top Recommendation: Ultralytics YOLO for Person Detection + Crop

Simplest approach that meets requirements. YOLO class 0 = "person". Fast, reliable, CPU-compatible.

### Installation

```bash
uv add ultralytics
```

### Usage

```python
from ultralytics import YOLO
from PIL import Image
import numpy as np

class PersonDetector:
    """Detect and crop people using YOLO."""

    def __init__(self, model_size: str = "n"):
        # Model sizes: n (nano), s (small), m (medium), l (large), x (xlarge)
        # nano is fastest, xlarge is most accurate
        self.model = YOLO(f"yolo11{model_size}.pt")

    def detect_people(self, image: Image.Image, conf: float = 0.5, padding: int = 10):
        """Detect people and return cropped regions."""
        results = self.model.predict(
            image,
            classes=[0],  # class 0 = person
            conf=conf,
            device="cpu",  # or "mps" for Apple Silicon
            verbose=False,
        )

        crops = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                # Add padding
                w, h = image.size
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)

                crop = image.crop((x1, y1, x2, y2))
                crops.append({
                    "image": crop,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(result.boxes.conf[0]),
                })

        return crops
```

### Performance (Apple M3)

- yolo11n: ~50ms per image
- yolo11s: ~80ms per image
- yolo11m: ~150ms per image

## Alternative 1: YOLOv8/YOLO11 Instance Segmentation

Provides pixel-level masks, not just bounding boxes. Slightly slower but better quality.

### Installation

```bash
uv add ultralytics
```

### Usage

```python
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

class PersonSegmenter:
    """Segment people with pixel-level masks using YOLO."""

    def __init__(self, model_size: str = "n"):
        self.model = YOLO(f"yolo11{model_size}-seg.pt")

    def segment_people(self, image: Image.Image, conf: float = 0.5, padding: int = 10):
        """Segment people and return crops with masks."""
        results = self.model.predict(
            image,
            classes=[0],  # person class
            conf=conf,
            device="cpu",  # or "mps"
            verbose=False,
        )

        crops = []
        for result in results:
            if result.masks is None:
                continue

            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()

            for mask, box in zip(masks, boxes):
                x1, y1, x2, y2 = map(int, box)

                # Resize mask to original image size
                h, w = image.size[1], image.size[0]
                mask_resized = cv2.resize(mask, (w, h))

                # Add padding
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)

                crop = image.crop((x1, y1, x2, y2))
                mask_crop = mask_resized[y1:y2, x1:x2]

                crops.append({
                    "image": crop,
                    "bbox": (x1, y1, x2, y2),
                    "mask": mask_crop,
                })

        return crops
```

### Performance (Apple M3)

- yolo11n-seg: ~100ms per image
- yolo11s-seg: ~150ms per image
- yolo11m-seg: ~250ms per image

## Alternative 2: MediaPipe Image Segmentation

Google's solution, optimized for mobile/edge. Very fast, good for single-person selfie-style images.

### Installation

```bash
uv add mediapipe
```

### Usage - Selfie Segmentation (Legacy API)

```python
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

class MediaPipeSegmenter:
    """Segment people using MediaPipe Selfie Segmentation."""

    def __init__(self, model_selection: int = 1):
        # model_selection: 0 = general, 1 = landscape (faster)
        self.mp_selfie = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie.SelfieSegmentation(
            model_selection=model_selection
        )

    def segment(self, image: Image.Image, threshold: float = 0.5):
        """Get person segmentation mask."""
        img_array = np.array(image)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

        results = self.segmenter.process(img_rgb)
        mask = results.segmentation_mask

        # Binary mask
        binary_mask = (mask > threshold).astype(np.uint8)

        return binary_mask

    def extract_person(self, image: Image.Image, padding: int = 10):
        """Extract person crop from image."""
        mask = self.segment(image)

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

        return {
            "image": image.crop((x1, y1, x2, y2)),
            "bbox": (x1, y1, x2, y2),
            "mask": mask,
        }
```

### Usage - New Tasks API (Multiclass)

```python
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image

class MediaPipeMulticlassSegmenter:
    """Multiclass segmentation: hair, face, body, clothes, etc."""

    def __init__(self, model_path: str = "selfie_multiclass_256x256.tflite"):
        # Download model from:
        # https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True,
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def segment(self, image: Image.Image):
        """Get multiclass segmentation.

        Categories:
        0: background
        1: hair
        2: body-skin
        3: face-skin
        4: clothes
        5: others (accessories)
        """
        mp_image = vision.Image(image_format=vision.ImageFormat.RGB, data=np.array(image))
        result = self.segmenter.segment(mp_image)
        category_mask = result.category_mask.numpy_view()
        return category_mask
```

### Performance (Apple M3)

- Selfie Segmentation: ~30-50ms per image
- Multiclass (256x256): ~40-60ms per image

## Alternative 3: FastSAM

CNN-based SAM alternative. 50x faster than original SAM, runs on CPU.

### Installation

```bash
uv add ultralytics
```

### Usage

```python
from ultralytics import FastSAM
from PIL import Image

class FastSAMSegmenter:
    """Segment using FastSAM with text or point prompts."""

    def __init__(self, model: str = "FastSAM-s.pt"):
        # FastSAM-s (small) or FastSAM-x (large)
        self.model = FastSAM(model)

    def segment_people(self, image: Image.Image):
        """Segment all people in image using text prompt."""
        results = self.model(
            image,
            device="cpu",  # or "mps"
            retina_masks=True,
            imgsz=640,
            conf=0.4,
            iou=0.9,
            texts="person",  # Text prompt
        )

        crops = []
        for result in results:
            if result.masks is None:
                continue

            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()

            for mask, box in zip(masks, boxes):
                x1, y1, x2, y2 = map(int, box)
                crop = image.crop((x1, y1, x2, y2))
                crops.append({
                    "image": crop,
                    "bbox": (x1, y1, x2, y2),
                    "mask": mask,
                })

        return crops
```

### Performance (Apple M3)

- FastSAM-s: ~200-400ms per image
- FastSAM-x: ~400-800ms per image

## Alternative 4: MobileSAM

Distilled SAM with tiny ViT encoder. High quality, reasonable speed.

### Installation

```bash
uv add ultralytics
```

### Usage

```python
from ultralytics import SAM
from PIL import Image

class MobileSAMSegmenter:
    """High-quality segmentation with MobileSAM."""

    def __init__(self):
        self.model = SAM("mobile_sam.pt")

    def segment_at_points(self, image: Image.Image, points: list[tuple[int, int]]):
        """Segment objects at specific points."""
        results = self.model.predict(
            image,
            points=points,
            labels=[1] * len(points),  # 1 = foreground
            device="cpu",
        )

        masks = []
        for result in results:
            if result.masks is not None:
                masks.extend(result.masks.data.cpu().numpy())

        return masks

    def segment_in_box(self, image: Image.Image, bbox: tuple[int, int, int, int]):
        """Segment object within bounding box."""
        results = self.model.predict(
            image,
            bboxes=[bbox],
            device="cpu",
        )

        if results[0].masks is not None:
            return results[0].masks.data.cpu().numpy()[0]
        return None
```

### Hybrid Approach: YOLO Detection + MobileSAM Refinement

```python
from ultralytics import YOLO, SAM
from PIL import Image

class HybridSegmenter:
    """Use YOLO for detection, MobileSAM for precise masks."""

    def __init__(self):
        self.detector = YOLO("yolo11n.pt")
        self.sam = SAM("mobile_sam.pt")

    def segment_people(self, image: Image.Image, conf: float = 0.5):
        """Detect people with YOLO, refine masks with SAM."""
        # Step 1: Fast detection
        detections = self.detector.predict(
            image,
            classes=[0],
            conf=conf,
            device="cpu",
            verbose=False,
        )

        crops = []
        for result in detections:
            boxes = result.boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                # Step 2: Precise segmentation with SAM
                sam_results = self.sam.predict(
                    image,
                    bboxes=[[x1, y1, x2, y2]],
                    device="cpu",
                )

                if sam_results[0].masks is not None:
                    mask = sam_results[0].masks.data.cpu().numpy()[0]
                    crop = image.crop((x1, y1, x2, y2))
                    crops.append({
                        "image": crop,
                        "bbox": (x1, y1, x2, y2),
                        "mask": mask,
                    })

        return crops
```

### Performance (Apple M3)

- MobileSAM alone: ~300-500ms per prompt
- YOLO + MobileSAM hybrid: ~400-700ms per image (depends on person count)

## Face-Specific Detection

For face-only crops, consider these options:

### MediaPipe Face Detection

```python
import mediapipe as mp
from PIL import Image
import numpy as np

class FaceDetector:
    def __init__(self):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=1,  # 0=short-range, 1=full-range
            min_detection_confidence=0.5,
        )

    def detect_faces(self, image: Image.Image, padding: int = 20):
        """Detect and crop faces."""
        img_rgb = np.array(image)
        results = self.detector.process(img_rgb)

        crops = []
        if results.detections:
            h, w = image.size[1], image.size[0]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w) - padding
                y1 = int(bbox.ymin * h) - padding
                x2 = int((bbox.xmin + bbox.width) * w) + padding
                y2 = int((bbox.ymin + bbox.height) * h) + padding

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                crops.append({
                    "image": image.crop((x1, y1, x2, y2)),
                    "bbox": (x1, y1, x2, y2),
                    "confidence": detection.score[0],
                })

        return crops
```

### YOLO Face Models

```bash
# Install yolov8-face or similar
uv add ultralytics
```

```python
from ultralytics import YOLO

# Use a face-specific model
model = YOLO("yolov8n-face.pt")  # Community model
```

## Implementation Recommendation for ai-detect

Given the project's needs (crop people/faces for AI detection analysis), I recommend:

### Option A: Simple and Fast (Recommended)

Use YOLO11n for person detection with bounding box crops. No segmentation needed if the goal is just to analyze different regions.

```python
# In segment.py, add fallback:
class PersonDetector:
    """CPU-compatible person detection using YOLO."""

    def __init__(self, device: str | None = None):
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self._model = None

    def load(self):
        from ultralytics import YOLO
        self._model = YOLO("yolo11n.pt")

    def detect_people(self, image: Image.Image, padding: int = 10) -> list[PersonCrop]:
        if self._model is None:
            self.load()

        results = self._model.predict(
            image,
            classes=[0],
            conf=0.5,
            device=self.device,
            verbose=False,
        )

        crops = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                w, h = image.size
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)

                crops.append(PersonCrop(
                    image=image.crop((x1, y1, x2, y2)),
                    bbox=(x1, y1, x2, y2),
                    mask=np.ones((y2-y1, x2-x1), dtype=np.uint8),  # Dummy mask
                ))

        return crops
```

### Option B: Quality Masks

Use YOLO11n-seg for instance segmentation with proper masks.

### Dependencies to Add

```bash
uv add ultralytics  # For YOLO/FastSAM/MobileSAM
uv add mediapipe    # For face detection (optional)
```

## References

- Ultralytics YOLO: https://docs.ultralytics.com/
- MediaPipe: https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter
- FastSAM: https://github.com/CASIA-LMC-Lab/FastSAM
- MobileSAM: https://github.com/ChaoningZhang/MobileSAM
