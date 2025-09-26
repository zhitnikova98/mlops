import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from pathlib import Path
import onnx
from PIL import Image
import requests


class BlipONNXConverter:
    """
    Конвертер модели BLIP в ONNX формат
    """

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        self.model_name = model_name
        self.processor = None
        self.model = None

    def load_model(self):
        """Загрузка модели и процессора"""
        print(f"Загрузка модели {self.model_name}")
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        self.model.eval()

    def convert_to_onnx(self, onnx_path: str = "models/blip_model.onnx"):
        """Конвертация модели в ONNX формат"""
        if self.model is None:
            raise ValueError("Модель не загружена. Вызовите load_model() сначала.")

        # Создание папки для модели
        Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

        # Создание dummy input для трассировки
        dummy_image = torch.randn(1, 3, 384, 384)
        dummy_input_ids = torch.randint(0, 1000, (1, 16))

        print("Конвертация в ONNX...")

        # Экспорт в ONNX
        torch.onnx.export(
            self.model,
            (dummy_image, dummy_input_ids),
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["image", "input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "image": {0: "batch_size"},
                "input_ids": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )

        print(f"Модель сохранена в {onnx_path}")

        # Проверка модели
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX модель прошла проверку")

        return onnx_path

    def test_pytorch_model(self, image_url: str = None):
        """Тестирование PyTorch модели на примере изображения"""
        if self.model is None or self.processor is None:
            raise ValueError("Модель не загружена. Вызовите load_model() сначала.")

        # Загрузка тестового изображения
        if image_url is None:
            image_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"

        print(f"Загрузка изображения: {image_url}")
        raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

        # Предобработка
        inputs = self.processor(raw_image, return_tensors="pt")

        # Генерация caption
        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=50)

        caption = self.processor.decode(out[0], skip_special_tokens=True)
        print(f"PyTorch модель caption: {caption}")

        return caption


def main():
    """Основная функция для конвертации и тестирования модели"""
    converter = BlipONNXConverter()

    # Загрузка модели
    converter.load_model()

    # Тестирование PyTorch модели
    converter.test_pytorch_model()

    # Конвертация в ONNX
    onnx_path = converter.convert_to_onnx()

    print("\nКонвертация завершена успешно!")
    print(f"ONNX модель сохранена в: {onnx_path}")


if __name__ == "__main__":
    main()
