from __future__ import annotations

import io
import os
import base64
import logging

import cv2
import timm
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.domain.entities.analysis_result import ModelMetrics
from src.domain.interfaces import IModelService

log = logging.getLogger("infrastructure.model_service")


class ModelService(IModelService):
    """
    PyTorch + timm SwinV2 deepfake detection service
    """

    MODEL_PATH = os.getenv(
        "MODEL_PATH",
        "models/best_swinv2.pth"
    )

    MODEL_NAME = "swinv2_tiny_window16_256.ms_in1k"

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        log.info("Model yükleniyor: %s", self.MODEL_PATH)

        # Model oluştur
        self.model = timm.create_model(
            self.MODEL_NAME,
            pretrained=False,
            num_classes=2
        )

        # Weight yükle
        state_dict = torch.load(
            self.MODEL_PATH,
            map_location=self.device
        )

        self.model.load_state_dict(state_dict)

        # Device taşı
        self.model.to(self.device)

        # Eval mode
        self.model.eval()

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        log.info("✔ SwinV2 model hazır")

    def analyze(self, image_bytes: bytes) -> ModelMetrics:
        # Bytes -> PIL
        pil_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        # Preprocess
        img_tensor = self.transform(pil_image)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]

        # probs[0] = REAL
        # probs[1] = FAKE
        fake_score = float(probs[1].item())

        is_deepfake = fake_score >= 0.5

        gradcam_b64 = self._simple_overlay(
            pil_image,
            fake_score
        )

        return ModelMetrics(
            is_deepfake=is_deepfake,
            confidence=round(fake_score, 4),
            gradcam_b64=gradcam_b64,
        )

    @staticmethod
    def _simple_overlay(
        pil_image: Image.Image,
        score: float
    ) -> str:

        img = pil_image.resize(
            (224, 224),
            Image.Resampling.LANCZOS
        )

        img_bgr = cv2.cvtColor(
            np.array(img),
            cv2.COLOR_RGB2BGR
        )

        intensity = int(score * 255)

        overlay = np.zeros_like(img_bgr)

        overlay[:, :, 2] = intensity
        overlay[:, :, 1] = 255 - intensity

        result = cv2.addWeighted(
            img_bgr,
            0.7,
            overlay,
            0.3,
            0
        )

        ok, buf = cv2.imencode(
            ".jpg",
            result,
            [cv2.IMWRITE_JPEG_QUALITY, 90]
        )

        if not ok:
            raise RuntimeError(
                "Overlay encode edilemedi"
            )

        return base64.b64encode(
            buf.tobytes()
        ).decode("utf-8")