"""
Infrastructure: ModelService
==============================
HuggingFace EfficientNet tabanlı deepfake tespiti.
Model: dima806/deepfake_vs_real_image_detection
"""
from __future__ import annotations

import io
import base64
import logging

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

from src.domain.entities.analysis_result import ModelMetrics
from src.domain.interfaces import IModelService

log = logging.getLogger("infrastructure.model_service")


class ModelService(IModelService):
    """
    HuggingFace pipeline ile deepfake tespiti.
    İlk çağrıda model indirilir ve cache'lenir (~50MB).
    Kendi modelinle değiştirmek için yalnızca bu sınıfı güncelle.
    """

    MODEL_ID = "dima806/deepfake_vs_real_image_detection"

    def __init__(self) -> None:
        log.info("Model yükleniyor: %s", self.MODEL_ID)
        self._pipe = pipeline(
            "image-classification",
            model=self.MODEL_ID,
        )
        log.info("✔ Model hazır")

    def analyze(self, image_bytes: bytes) -> ModelMetrics:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Model tahmini
        results = self._pipe(pil_image)

        # Sonuçları parse et
        # Etiketler: "Fake" / "Real"
        scores = {r["label"].lower(): r["score"] for r in results}
        fake_score = scores.get("fake", scores.get("deepfake", 0.0))
        is_deepfake = fake_score >= 0.5

        # Grad-CAM yerine basit highlight (kendi modelinde gerçek Grad-CAM eklenebilir)
        gradcam_b64 = self._simple_overlay(pil_image, fake_score)

        return ModelMetrics(
            is_deepfake = is_deepfake,
            confidence  = round(fake_score, 4),
            gradcam_b64 = gradcam_b64,
        )

    # ── Yardımcılar ───────────────────────────────────────────────

    @staticmethod
    def _simple_overlay(pil_image: Image.Image, score: float) -> str:
        """
        Sahtelik skoruna göre kırmızı/yeşil overlay.
        Gerçek Grad-CAM kendi modelini eğitince eklenebilir.
        """
        img = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Yüksek fake skoru → kırmızıya yakın, düşük → yeşile yakın
        intensity = int(score * 255)
        overlay   = np.zeros_like(img_bgr)
        overlay[:, :, 2] = intensity          # R kanalı (BGR)
        overlay[:, :, 1] = 255 - intensity    # G kanalı

        result = cv2.addWeighted(img_bgr, 0.7, overlay, 0.3, 0)

        ok, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("Overlay encode edilemedi")
        return base64.b64encode(buf.tobytes()).decode("utf-8")