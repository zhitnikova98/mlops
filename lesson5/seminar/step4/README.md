# Step 4: Active Learning vs Full Dataset Comparison

Этот шаг демонстрирует сравнение эффективности Active Learning с обучением на полном датасете.

## Основные возможности

### 🎯 **Новый датасет**
- **Digits Dataset**: Классификация рукописных цифр (0-9)
- 1797 образцов, 64 признака (8x8 пикселей)
- 10 классов (цифры 0-9)
- Более сложная задача по сравнению с Forest Cover Type

### 🔄 **Active Learning Pipeline**
- **Uncertainty Sampling**: 3 стратегии (entropy, margin, least confident)
- **Incremental Learning**: Постепенное добавление данных
- **MLflow Tracking**: Отслеживание всех экспериментов
- **Prefect Orchestration**: Управление workflow

### 📊 **Baseline Comparison**
- **Full Dataset Training**: Обучение на всем доступном датасете
- **Performance Metrics**: Сравнение accuracy, F1-score
- **Data Efficiency**: Анализ количества данных vs качество модели
- **Learning Curves**: Визуализация процесса обучения

## Архитектура

```
step4/
├── src/
│   ├── data_manager.py          # Управление данными (digits dataset)
│   ├── model_trainer.py         # Обучение CatBoost
│   ├── active_learning.py       # AL стратегии
│   └── baseline_trainer.py      # NEW: Обучение на полном датасете
├── flows/
│   ├── active_learning_flow.py  # AL Prefect flow
│   └── comparison_flow.py       # NEW: Сравнение AL vs Baseline
├── pyproject.toml              # Poetry конфигурация
├── Makefile                    # Автоматизация
└── README.md                   # Документация
```

## Использование

### 🚀 **Быстрый старт**
```bash
# Установка зависимостей
make install

# Запуск сервисов
make start-services

# Сравнение AL vs Baseline
make compare-all
```

### 🔬 **Отдельные эксперименты**
```bash
# Только Active Learning (entropy)
make run-al-entropy

# Только Baseline (полный датасет)
make run-baseline

# Сравнение всех стратегий
make benchmark
```

### 📈 **Анализ результатов**
```bash
# MLflow UI
make mlflow-ui
# Откройте http://localhost:5000

# Prefect UI
make prefect-server
# Откройте http://localhost:4200
```

## Ожидаемые результаты

### 🎯 **Active Learning**
- **Эффективность данных**: Достижение 85-90% качества с 30-50% данных
- **Uncertainty Sampling**: Entropy обычно показывает лучшие результаты
- **Convergence**: Быстрая сходимость на первых итерациях

### 📊 **Baseline (Full Dataset)**
- **Maximum Performance**: Верхняя граница качества модели
- **Data Utilization**: 100% использование данных
- **Reference Point**: Эталон для сравнения AL стратегий

### 🔍 **Сравнение**
- **Data Efficiency**: AL достигает 90% качества baseline с 40-60% данных
- **Training Speed**: AL быстрее на ранних стадиях
- **Practical Value**: AL особенно эффективен при ограниченных ресурсах разметки

## Команды Makefile

| Команда | Описание |
|---------|----------|
| `make install` | Установка зависимостей |
| `make start-services` | Запуск Prefect + MLflow |
| `make run-al-entropy` | AL с entropy sampling |
| `make run-al-margin` | AL с margin sampling |
| `make run-al-confident` | AL с least confident sampling |
| `make run-baseline` | Обучение на полном датасете |
| `make compare-all` | Полное сравнение всех методов |
| `make benchmark` | Бенчмарк всех AL стратегий |
| `make clean` | Очистка артефактов |
| `make stop-services` | Остановка сервисов |

## Технические детали

### 🔧 **Конфигурация**
- **Python**: 3.9+
- **CatBoost**: Gradient boosting для классификации
- **MLflow**: Experiment tracking
- **Prefect**: Workflow orchestration
- **Poetry**: Dependency management

### 📝 **Логирование**
- **MLflow Experiments**: Отдельные эксперименты для AL и Baseline
- **Metrics**: accuracy, f1_macro, f1_weighted, precision, recall
- **Parameters**: sampling_strategy, batch_size, iterations
- **Artifacts**: Модели, confusion matrices, learning curves

### 🎛️ **Настройки**
- **Initial Labeled**: 10% от training set
- **Batch Size**: 10% от remaining pool
- **Max Iterations**: 10 (до исчерпания pool)
- **Random Seed**: 42 (воспроизводимость)
