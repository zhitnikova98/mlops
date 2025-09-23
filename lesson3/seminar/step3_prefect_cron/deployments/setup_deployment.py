"""
Настройка деплойментов для автоматического пайплайна.
"""

import sys
import os

# Добавляем flows в путь
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from flows.automated_training_flow import automated_training_pipeline


def create_automated_deployment():
    """Создание деплоймента с cron расписанием."""

    deployment = Deployment.build_from_flow(
        flow=automated_training_pipeline,
        name="automated-ml-pipeline",
        description="Автоматический ML пайплайн, запускающийся каждые 2 минуты",
        version="1.0.0",
        schedule=CronSchedule(cron="*/2 * * * *", timezone="UTC"),  # каждые 2 минуты
        work_queue_name="default",
        parameters={},
        tags=["ml", "automated", "cron"],
    )

    return deployment


def create_manual_deployment():
    """Создание деплоймента для ручного запуска."""
    from flows.automated_training_flow import manual_training_pipeline

    deployment = Deployment.build_from_flow(
        flow=manual_training_pipeline,
        name="manual-ml-pipeline",
        description="Ручной запуск ML пайплайна для конкретного батча",
        version="1.0.0",
        work_queue_name="default",
        tags=["ml", "manual"],
    )

    return deployment


if __name__ == "__main__":
    print("Создание деплойментов...")

    # Создаем автоматический деплоймент
    auto_deployment = create_automated_deployment()
    auto_deployment_id = auto_deployment.apply()
    print(f"Создан автоматический деплоймент: {auto_deployment_id}")

    # Создаем ручной деплоймент
    manual_deployment = create_manual_deployment()
    manual_deployment_id = manual_deployment.apply()
    print(f"Создан ручной деплоймент: {manual_deployment_id}")

    print("\nДеплойменты созданы!")
    print("Запустите Prefect agent для выполнения:")
    print("prefect agent start -q default")
