import logging
import json
import sys
import os

class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',      # Синий
        'INFO': '\033[92m',       # Зелёный
        'WARNING': '\033[93m',    # Жёлтый
        'ERROR': '\033[91m',      # Красный
        'CRITICAL': '\033[1;91m'  # Жирный красный
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage()
        }
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_record, ensure_ascii=False)

class MetricsFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return 'metric' in msg.lower() and record.levelno == logging.INFO

def get_logger(name=__name__, console_colors=True, log_file=None, metrics_file=None):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    # Автоматическое создание папок для логов
    for path in [log_file, metrics_file]:
        if path:
            folder = os.path.dirname(path)
            if folder and not os.path.exists(folder):
                os.makedirs(folder)

    # Консольный вывод
    console_handler = logging.StreamHandler(sys.stdout)
    if console_colors:
        formatter = ColorFormatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
    else:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Файл с JSON-логами общего назначения
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(JsonFormatter(datefmt='%Y-%m-%dT%H:%M:%S'))
        logger.addHandler(file_handler)

    # Файл для метрик в формате JSONL
    if metrics_file:
        metrics_handler = logging.FileHandler(metrics_file, mode='a', encoding='utf-8')
        metrics_handler.setFormatter(JsonFormatter(datefmt='%Y-%m-%dT%H:%M:%S'))
        metrics_handler.addFilter(MetricsFilter())
        logger.addHandler(metrics_handler)

    return logger