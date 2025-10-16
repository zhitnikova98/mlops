import time
from monitor import ApiMonitor
from config import get_config
from logger import get_logger

config = get_config()

logger = get_logger(
    'monitor',
    console_colors=config['logging']['console_colors'],
    log_file=config['logging']['log_file'],
    metrics_file=config['logging']['metrics_file']
)

API_URL = config['service']['base_url']
IMG_FOLDER = './test_images'
CHECK_INTERVAL = config['monitoring']['check_interval_seconds']
SAMPLES_PER_CHECK = config['monitoring']['samples_per_check']

monitor = ApiMonitor(API_URL, IMG_FOLDER, logger, config)

for _ in range(3):
    monitor.run_monitor()
    # time.sleep(CHECK_INTERVAL)
