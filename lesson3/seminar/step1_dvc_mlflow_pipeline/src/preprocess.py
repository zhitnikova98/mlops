"""
Предобработка данных для обучения модели.
"""

import os
import sys
import pandas as pd


def preprocess_data(batch_number: int):
    """
    Предобработка данных для обучения.

    Args:
        batch_number: Номер версии датасета
    """
    # Загружаем объединенные данные
    df = pd.read_csv(f"data/processed/dataset_v{batch_number}.csv")

    # Создаем целевую переменную - высокие чаевые (выше медианы)
    df["high_tip"] = (df["tip"] > df["tip"].median()).astype(int)

    # Оставляем только нужные колонки для обучения
    features = ["total_bill", "size", "high_tip"]
    df_processed = df[features].copy()

    # Убираем пропуски
    df_processed = df_processed.dropna()

    # Создаем папку если её нет
    os.makedirs("data/processed", exist_ok=True)

    # Сохраняем обработанные данные
    df_processed.to_csv(
        f"data/processed/dataset_processed_v{batch_number}.csv", index=False
    )

    print(f"Предобработка завершена. Обработано {len(df_processed)} записей")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py <batch_number>")
        sys.exit(1)

    batch_num = int(sys.argv[1])
    preprocess_data(batch_num)
