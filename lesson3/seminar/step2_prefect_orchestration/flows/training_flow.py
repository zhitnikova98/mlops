"""
Основной поток обучения ML модели в Prefect.
"""

import sys
import os

# Добавляем src в путь для импорта
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from prefect import flow
from data_tasks import (
    load_params,
    get_raw_data,
    prepare_batch,
    merge_batches,
    preprocess_data,
    create_dvc_tracking_file,
)
from model_tasks import train_model, evaluate_model


@flow(name="ML Training Pipeline", log_prints=True)
def training_pipeline(batch_number: int = 1):
    """
    Основной поток ML пайплайна с явными связями между задачами.

    Args:
        batch_number: Номер батча для обучения
    """
    print(f"Запуск пайплайна для батча {batch_number}")

    # Загружаем параметры
    params = load_params()

    # Получаем исходные данные (только если их нет)
    raw_data_path = "data/raw/tips_full.csv"
    if not os.path.exists(raw_data_path):
        get_raw_data(params["data"]["url"])

    # Подготавливаем текущий батч
    batch_size = prepare_batch(batch_number, params["data"]["batch_size"])

    if batch_size == 0:
        print("Нет данных для обработки")
        return

    # Объединяем все батчи (зависит от batch_size)
    dataset_size = merge_batches(batch_number, batch_size)

    # Предобрабатываем данные (зависит от dataset_size)
    processed_size = preprocess_data(batch_number, dataset_size)

    # Создаем файл для DVC трекинга (зависит от processed_size)
    dvc_tracking = create_dvc_tracking_file(processed_size)

    # Обучаем модель (зависит от processed_size и dvc_tracking)
    train_metrics = train_model(batch_number, params, processed_size, dvc_tracking)

    # Оцениваем модель (зависит от train_metrics)
    eval_metrics = evaluate_model(batch_number, params, train_metrics)

    print(f"Пайплайн завершен для батча {batch_number}")
    print(f"Обработано данных: {processed_size}")
    print(f"Метрики обучения: {train_metrics}")
    print(f"Метрики тестирования: {eval_metrics}")

    return {
        "batch_number": batch_number,
        "processed_size": processed_size,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        batch_num = int(sys.argv[1])
    else:
        batch_num = 1

    training_pipeline(batch_num)
