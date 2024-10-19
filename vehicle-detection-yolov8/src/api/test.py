import cv2
import numpy as np

import rootutils

ROOT = rootutils.autosetup()

from io import BytesIO
from typing import List

import cv2
import numpy as np
import uvicorn
from fastapi import APIRouter, Depends, FastAPI
from fastapi.responses import Response
from PIL import Image

from src.engine.yolo_onnx_engine import YoloOnnxEngine
from src.schema.api_schema import SnapshotRequestSchema, SnapshotResponseSchema

class TestRun:
    def __init__(
        self,
        engine_path: str,
        categories: List[str] = ["car", "truck", "bus", "pedestrian"],
        provider: str = "gpu",
        conf = 0.25
    ) -> None:
        """Initialize Snapshot API module."""
        self.engine_path = engine_path
        self.categories = categories
        self.provider = provider
        self.conf = conf

        self.setup_engine()

    def snapshot(
        self, 
        img: np.ndarray,  # RGB image as input in NumPy array format
    ):
        """Detect objects in an RGB image."""

        print(f"Processing snapshot...")

        # Assuming the image is already RGB, no need for preprocessing
        # detect objects
        dets = self.engine.detect(img, conf=self.conf)[0]

        # convert to result schema
        result: List[SnapshotResponseSchema] = []
        for box, score, cat in zip(dets.boxes, dets.scores, dets.categories):
            result.append(
                SnapshotResponseSchema(
                    category=cat,
                    box=box,
                    score=score,
                )
            )

        for res in result:
            if (res.box[1]+res.box[3])/2 > (-(res.box[0]+res.box[2])/2*3/5 + 230):
                cv2.rectangle(
                    img,
                    (res.box[0], res.box[1]),
                    (res.box[2], res.box[3]),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    img,
                    f"{res.category} {res.score:.2f}",
                    (res.box[0], res.box[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            # return image
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("./out.png", img)

        return result

    def setup_engine(self) -> None:
        """Setup YOLOv8 ONNX engine."""
        self.engine = YoloOnnxEngine(
            engine_path=self.engine_path,
            categories=self.categories,
            provider=self.provider,
        )
        self.engine.setup()

    # Example usage
    def read_image_with_cv2(self, file_path: str) -> np.ndarray:
        """Reads an image from file using OpenCV and converts it to RGB."""
        # Read the image using OpenCV (default is BGR format)
        img_bgr = cv2.imread(file_path)
        
        # Convert the BGR image to RGB format
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return img_rgb


# Usage example
file_path = "1.png"
conf_threshold = 0.15  # Example confidence threshold

T = TestRun("../../tmp/best.onnx", conf=conf_threshold)
img_rgb = T.read_image_with_cv2(file_path)

# Call the snapshot function asynchronously with the RGB image
result = T.snapshot(img_rgb)

# Process the detection results
print(result)
