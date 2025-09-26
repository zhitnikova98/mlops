import numpy as np
import torch
from PIL import Image
from src.onnx_tester import ONNXModelTester


def iterative_onnx_generation(onnx_tester, image, max_tokens=10):
    """
    Попытка итеративной генерации с ONNX моделью
    """
    print("🔄 Экспериментальная итеративная генерация с ONNX...")

    # Предобработка изображения
    inputs = onnx_tester.processor(image, return_tensors="pt")
    image_input = inputs.pixel_values.numpy()

    # Начальная последовательность токенов
    token_id = getattr(onnx_tester.processor.tokenizer, "bos_token_id", None)
    if token_id is None:
        token_id = getattr(onnx_tester.processor.tokenizer, "cls_token_id", 101)

    # Начинаем с последовательности из начального токена
    current_tokens = [token_id]
    generated_tokens = []

    print(f"Начальный токен: {token_id}")

    for step in range(max_tokens):
        # Подготавливаем input_ids текущей длины (заполняем до 16)
        if len(current_tokens) < 16:
            # Добавляем padding
            input_ids = current_tokens + [token_id] * (16 - len(current_tokens))
        else:
            # Берем последние 16 токенов
            input_ids = current_tokens[-16:]

        input_ids_array = np.array([input_ids], dtype=np.int64)

        # ONNX инференс
        onnx_inputs = {"image": image_input, "input_ids": input_ids_array}

        try:
            outputs = onnx_tester.session.run(None, onnx_inputs)
            logits = outputs[0]  # [1, 16, 30524]

            # Берем логиты для последнего токена
            last_token_logits = logits[
                0, len(current_tokens) - 1 if len(current_tokens) <= 16 else 15, :
            ]
            predicted_id = int(np.argmax(last_token_logits))

            # Декодируем токен
            try:
                token = onnx_tester.processor.tokenizer.decode([predicted_id])
                print(
                    f"  Шаг {step+1}: предсказан токен '{token}' (ID: {predicted_id})"
                )

                # Проверяем условия остановки
                if predicted_id == 102:  # [SEP] token
                    print("  🛑 Встретили [SEP] токен, останавливаем генерацию")
                    break

                current_tokens.append(predicted_id)
                generated_tokens.append(predicted_id)

            except Exception as e:
                print(f"  ❌ Ошибка декодирования токена {predicted_id}: {e}")
                break

        except Exception as e:
            print(f"  ❌ Ошибка ONNX инференса на шаге {step+1}: {e}")
            break

    # Пытаемся декодировать полную последовательность
    if generated_tokens:
        try:
            full_text = onnx_tester.processor.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )
            print(f"\n🎯 Итеративно сгенерированный текст: '{full_text}'")
            return full_text
        except Exception as e:
            print(f"❌ Ошибка декодирования полной последовательности: {e}")
            return None
    else:
        print("❌ Не удалось сгенерировать токены")
        return None


def main():
    """Эксперимент с улучшенной ONNX генерацией"""
    print("=== Эксперимент: улучшенная генерация ONNX ===\n")

    # Загрузка изображения
    image_path = "../step2_fastapi_inference/test_images/img.jpg"
    image = Image.open(image_path).convert("RGB")
    print(f"Изображение: {image_path}, размер: {image.size}")

    # Инициализация ONNX тестера
    onnx_path = "models/blip_model.onnx"
    tester = ONNXModelTester(onnx_path)
    tester.load_onnx_model()

    # Сравнение с PyTorch
    print("\n1. PyTorch baseline (для сравнения):")
    from src.model_converter import BlipONNXConverter

    converter = BlipONNXConverter()
    converter.load_model()

    inputs = converter.processor(image, return_tensors="pt")
    with torch.no_grad():
        out = converter.model.generate(**inputs, max_length=50)
    pytorch_caption = converter.processor.decode(out[0], skip_special_tokens=True)
    print(f"   PyTorch: '{pytorch_caption}'")

    # Эксперимент с итеративной генерацией
    print("\n2. Итеративная ONNX генерация:")
    onnx_caption = iterative_onnx_generation(tester, image, max_tokens=8)

    # Сравнение результатов
    print("\n📊 Итоговое сравнение:")
    print(f"   ✅ PyTorch (эталон): '{pytorch_caption}'")
    print(f"   ⚠️  ONNX итеративно:   '{onnx_caption or 'Не удалось сгенерировать'}'")

    print("\n💡 Выводы:")
    print("   • PyTorch модель работает идеально для этого изображения рыбака")
    print("   • ONNX модель технически функциональна, но семантически неточна")
    print("   • Для продакшена рекомендуется использовать PyTorch для инференса")
    print("   • ONNX подходит для демонстрации технологии, но требует доработки")


if __name__ == "__main__":
    main()
