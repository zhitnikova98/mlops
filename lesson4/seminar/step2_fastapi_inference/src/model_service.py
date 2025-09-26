import onnxruntime as ort
import numpy as np
from PIL import Image
from transformers import BlipProcessor
import torch
from typing import Optional
import time


class ONNXImageCaptionService:
    """
    Сервис для инференса ONNX модели image captioning
    """

    def __init__(
        self, onnx_path: str, model_name: str = "Salesforce/blip-image-captioning-base"
    ):
        self.onnx_path = onnx_path
        self.model_name = model_name
        self.session: Optional[ort.InferenceSession] = None
        self.processor: Optional[BlipProcessor] = None
        self.loaded = False

    def load_model(self):
        """Загрузка ONNX модели и процессора"""
        if self.loaded:
            return

        print(f"Загрузка ONNX модели из {self.onnx_path}")

        # Создание сессии ONNX Runtime для CPU
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.onnx_path, providers=providers)

        # Загрузка процессора для предобработки
        self.processor = BlipProcessor.from_pretrained(self.model_name)

        self.loaded = True
        print("Модель и процессор загружены успешно")

    def preprocess_image(self, image: Image.Image) -> dict:
        """Предобработка изображения для модели"""
        # Конвертация в RGB если нужно
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Предобработка через процессор BLIP
        inputs = self.processor(image, return_tensors="pt")

        return {
            "image": inputs.pixel_values.numpy(),
            "input_ids": torch.tensor(
                [[self.processor.tokenizer.bos_token_id]]
            ).numpy(),
        }

    def predict(self, image: Image.Image) -> dict:
        """
        Выполнение инференса для одного изображения

        Args:
            image: PIL изображение

        Returns:
            dict с результатами инференса
        """
        if not self.loaded:
            raise ValueError("Модель не загружена. Вызовите load_model() сначала.")

        start_time = time.time()

        # Предобработка
        preprocess_start = time.time()
        inputs = self.preprocess_image(image)
        preprocess_time = time.time() - preprocess_start

        # Инференс
        inference_start = time.time()
        outputs = self.session.run(None, inputs)
        inference_time = time.time() - inference_start

        # Постобработка
        postprocess_start = time.time()
        logits = outputs[0]

        # Для простоты возвращаем сырые логиты
        # В реальности нужна более сложная генерация текста
        prediction_confidence = float(np.max(logits))
        postprocess_time = time.time() - postprocess_start

        total_time = time.time() - start_time

        return {
            "prediction": f"Generated caption (confidence: {prediction_confidence:.3f})",
            "logits_shape": list(logits.shape),
            "timing": {
                "total_ms": total_time * 1000,
                "preprocess_ms": preprocess_time * 1000,
                "inference_ms": inference_time * 1000,
                "postprocess_ms": postprocess_time * 1000,
            },
        }

    def predict_batch(self, images: list[Image.Image]) -> list[dict]:
        """
        Batch инференс (пока не поддерживается, выполняется последовательно)

        Args:
            images: Список PIL изображений

        Returns:
            Список результатов инференса
        """
        results = []

        batch_start = time.time()
        for i, image in enumerate(images):
            result = self.predict(image)
            result["batch_index"] = i
            results.append(result)

        batch_time = time.time() - batch_start

        # Добавляем статистику батча
        batch_stats = {
            "batch_size": len(images),
            "total_batch_time_ms": batch_time * 1000,
            "avg_time_per_image_ms": batch_time * 1000 / len(images) if images else 0,
            "total_inference_time_ms": sum(
                r["timing"]["inference_ms"] for r in results
            ),
        }

        return results, batch_stats
