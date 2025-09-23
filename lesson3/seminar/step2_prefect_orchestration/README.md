# Шаг 2: Prefect для оркестрации пайплайна

## Описание

В этом шаге мы переходим от использования DVC как основного оркестратора к Prefect. DVC используется только для версионирования данных, а вся логика выполнения пайплайна описана в Prefect flows и tasks.

## Структура

```
step2_prefect_orchestration/
├── data/              # Данные (версионируются DVC)
├── models/            # Модели
├── metrics/           # Метрики
├── src/              # Задачи Prefect
│   ├── data_tasks.py     # Задачи для работы с данными
│   └── model_tasks.py    # Задачи для работы с моделями
├── flows/            # Потоки Prefect
│   └── training_flow.py  # Основной поток обучения
└── dvc.yaml          # Только для версионирования
```

## Установка и настройка

1. Установка зависимостей:
```bash
poetry install
```

2. Активация виртуального окружения:
```bash
poetry shell
```

3. Запуск Prefect server (в отдельном терминале):
```bash
prefect server start
```

4. Запуск MLflow UI (в отдельном терминале):
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

5. Открыть Prefect UI: http://localhost:4200

## Использование

### Запуск пайплайна через Python

```bash
# Обучение первой модели
python flows/training_flow.py 1

# Обучение второй модели с дополнительными данными
python flows/training_flow.py 2

# Обучение третьей модели
python flows/training_flow.py 3
```

### Запуск пайплайна через Prefect CLI

```bash
# Регистрация потока
prefect deployment build flows/training_flow.py:training_pipeline -n "ml-pipeline"

# Применение деплоймента
prefect deployment apply training_pipeline-deployment.yaml

# Запуск конкретного батча
prefect deployment run "ML Training Pipeline/ml-pipeline" --param batch_number=1
```

### Версионирование данных с DVC

```bash
# Инициализация DVC (если еще не сделано)
dvc init --no-scm

# Отслеживание версий данных
dvc add data/processed/

# Коммит изменений в данных
dvc commit
```

## Преимущества Prefect

1. **Визуализация**: Графическое отображение пайплайна в UI
2. **Мониторинг**: Отслеживание статуса выполнения задач
3. **Обработка ошибок**: Автоматические повторы и обработка сбоев
4. **Параллелизм**: Возможность параллельного выполнения задач
5. **Планирование**: Встроенная поддержка cron-подобного планирования

## Архитектура

- **Tasks**: Атомарные функции для выполнения конкретных операций
- **Flows**: Композиция задач в логическую последовательность
- **Deployments**: Конфигурация для запуска потоков
- **Work Queues**: Очереди для распределения работы

## Мониторинг

1. **Prefect UI**: http://localhost:4200 - статус выполнения пайплайнов
2. **MLflow UI**: http://localhost:5000 - эксперименты и модели
3. **Логи**: Детальные логи выполнения в Prefect UI
