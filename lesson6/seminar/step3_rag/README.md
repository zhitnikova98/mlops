# Homework: RAG-based Animation System

## Задание

Реализовать систему создания анимаций из текстовых описаний с использованием RAG.

## Требования

### 1. Компоненты системы

- **LLM Client**: подключение к Ollama (qwen2.5:1.5b)
- **RAG Retriever**: поиск поз по описанию (TF-IDF, cosine similarity)
- **Animation Agent**: интеграция LLM + RAG + Pose API
- **Jupyter Notebook**: интерактивная демонстрация

### 2. Pipeline

```
User Input (text)
  → LLM (generate movement descriptions)
  → RAG (find similar poses in DB)
  → Pose API (visualize poses)
  → GIF Generator (create animation)
```

### 3. База данных

`poses_database.json` - 96 поз с описаниями:
```json
{
  "pose": {
    "Torso": [0, 0],
    "Head": [0, 60],
    "RH": [30, 35],
    "LH": [-30, 35],
    "RK": [15, -50],
    "LK": [-15, -50]
  },
  "description": "Руки вытянуты вперед на уровне плеч, ноги на ширине плеч"
}
```

### 4. Реализация

#### PoseRetriever
- Загрузка poses_database.json
- TF-IDF векторизация описаний
- Поиск по similarity (cosine_similarity)
- Метод `search(query, top_k=5)` → список поз с similarity

#### RAGAnimationAgent
- Метод `generate_movement_sequence(action, num_steps)` → LLM генерирует описания
- Метод `create_animation(action, num_steps, duration)` → полный pipeline
- Возврат: base64 encoded GIF + metadata

#### Notebook
- Инициализация агента
- 3 примера: простой танец, макарена, свой вариант
- Анализ similarity scores
- Визуализация результатов

### 5. Запуск

```bash
# Установка
poetry install

# Запуск сервисов
# Terminal 1: Ollama
docker-compose up ollama

# Terminal 2: Pose API (из step2)
cd ../step2_function_calling
docker-compose up pose_api

# Terminal 3: Notebook
make notebook
```

### 6. Критерии оценки

- ✅ RAG retriever работает (similarity > 0.7 для релевантных запросов)
- ✅ LLM генерирует корректные описания движений
- ✅ Pipeline создает GIF из текста
- ✅ Notebook с 3+ примерами
- ✅ Код чистый, без избыточных комментариев

### 7. Примеры запросов

- "танец макарена"
- "простой танец руками"
- "прыжок с вращением"
- "зарядка"
- "приветствие"

## Ожидаемый результат

- `src/pose_retriever.py` (~60-80 строк)
- `src/rag_agent.py` (~150-180 строк)
- `notebooks/rag_demo.ipynb` (10-15 ячеек)
- GIF анимации в `output/`

## Подсказки

1. Используйте `TfidfVectorizer` из sklearn
2. LLM промпт должен просить генерировать только описания поз
3. Обрабатывайте вывод LLM (удаляйте нумерацию)
4. Pose API возвращает base64 изображения
5. Pillow для создания GIF: `save(..., format='GIF', save_all=True, loop=0)`

## Зависимости

```toml
[tool.poetry.dependencies]
python = "^3.11"
openai = "^1.54.5"
requests = "^2.32.3"
pillow = "^11.0.0"
scikit-learn = "^1.5.2"
jupyter = "^1.1.1"
```
