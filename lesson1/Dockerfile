FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=file:./mlruns

WORKDIR /app

# Install Poetry
RUN pip install poetry==1.8.3
RUN poetry config virtualenvs.create false

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-interaction --no-ansi

# Copy source code and configs
COPY src src
COPY configs configs

CMD ["python", "-m", "src.app.train"]
