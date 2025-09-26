from src.model_converter import BlipONNXConverter
from src.onnx_tester import ONNXModelTester
from PIL import Image
import os
import torch
import numpy as np


def test_with_real_image():
    """
    Тестирование с реальным изображением
    """
    print("=== Тест с реальным изображением ===\n")

    # Путь к изображению
    image_path = "../step2_fastapi_inference/test_images/img.jpg"

    if not os.path.exists(image_path):
        print(f"❌ Изображение не найдено: {image_path}")
        return

    print(f"✅ Найдено изображение: {image_path}")

    # Загрузка и показ информации об изображении
    image = Image.open(image_path)
    print(f"Размер изображения: {image.size}")
    print(f"Режим: {image.mode}")

    # Тест PyTorch модели с реальным изображением
    print("\n1. Тестирование PyTorch модели:")
    converter = BlipONNXConverter()
    converter.load_model()

    # Предобработка изображения
    inputs = converter.processor(image, return_tensors="pt")
    print(f"Размер после предобработки: {inputs.pixel_values.shape}")

    # Генерация caption PyTorch моделью

    with torch.no_grad():
        out = converter.model.generate(**inputs, max_length=50)

    caption = converter.processor.decode(out[0], skip_special_tokens=True)
    print(f"PyTorch модель caption: '{caption}'")

    # Проверка ONNX модели
    print("\n2. Проверка ONNX модели:")
    onnx_path = "models/blip_model.onnx"

    if not os.path.exists(onnx_path):
        print("❌ ONNX модель не найдена, создаем...")
        converter.convert_to_onnx(onnx_path)

    # Тест ONNX с реальным изображением
    tester = ONNXModelTester(onnx_path)
    tester.load_onnx_model()

    # Простой тест загрузки изображения в ONNX
    try:
        inputs_onnx = converter.processor(image, return_tensors="pt")
        image_input = inputs_onnx.pixel_values.numpy()
        # Правильный token_id для BLIP (исправленная версия)
        token_id = getattr(converter.processor.tokenizer, "bos_token_id", None)
        if token_id is None:
            token_id = getattr(converter.processor.tokenizer, "cls_token_id", 101)

        print(f"Используем token_id: {token_id}")
        input_ids = np.array(
            [[token_id] * 16], dtype=np.int64
        )  # Правильный размер [1, 16]

        print("Входные данные для ONNX:")
        print(f"  - image: {image_input.shape} {image_input.dtype}")
        print(f"  - input_ids: {input_ids.shape} {input_ids.dtype}")

        # ONNX инференс с правильными входами
        onnx_inputs = {"image": image_input, "input_ids": input_ids}

        print("\n🚀 Запуск ONNX инференса...")
        outputs = tester.session.run(None, onnx_inputs)
        print(f"✅ ONNX работает! Количество выходов: {len(outputs)}")

        # Анализ выходов
        for i, output in enumerate(outputs):
            print(f"  Выход {i}: {output.shape}")

        # Декодирование первого выхода (логиты)
        logits = outputs[0]  # [1, 16, 30524]
        if len(logits.shape) == 3 and logits.shape[2] == 30524:
            # Берем последний токен из последовательности
            last_token_logits = logits[0, -1, :]  # [30524]
            predicted_id = int(np.argmax(last_token_logits))

            # Декодируем токен
            try:
                predicted_token = converter.processor.tokenizer.decode([predicted_id])
                print(
                    f"\n🎯 ONNX предсказанный токен: '{predicted_token}' (ID: {predicted_id})"
                )

                # Получаем топ-5 предсказаний для анализа
                top5_ids = np.argsort(last_token_logits)[-5:][::-1]
                print("Топ-5 предсказанных токенов:")
                for rank, tid in enumerate(top5_ids, 1):
                    try:
                        token = converter.processor.tokenizer.decode([tid])
                        prob = float(last_token_logits[tid])
                        print(f"  {rank}. '{token}' (ID: {tid}, логит: {prob:.2f})")
                    except Exception:
                        print(f"  {rank}. <token_{tid}> (ID: {tid})")

            except Exception as decode_error:
                print(f"Не удалось декодировать токен {predicted_id}: {decode_error}")

        print("\n📊 Сравнение результатов:")
        print(f"✅ PyTorch полное описание: '{caption}'")
        print(f"⚠️  ONNX предсказанный токен: '{predicted_token}' (только один токен)")
        print("\n💡 Объяснение:")
        print("   PyTorch использует полную генерацию текста (autoregressive)")
        print("   ONNX экспортирует только один шаг предсказания токена")
        print("   Для полной генерации нужно многократно вызывать ONNX модель")

    except Exception as e:
        print(f"❌ Ошибка подготовки данных для ONNX: {e}")

    print("\n✅ Тестирование завершено!")
    print("📸 PyTorch модель успешно обработала изображение")
    print(f"💬 Сгенерированное описание: '{caption}'")


if __name__ == "__main__":
    test_with_real_image()
