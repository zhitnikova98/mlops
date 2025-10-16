import requests
import time
import os
import glob

class ApiMonitor:
    def __init__(self, api_url, img_folder, logger, config):
        self.api_url = api_url
        self.img_folder = img_folder
        self.logger = logger
        self.config = config

        self.latencies_health = []
        self.latencies_predict = []
        self.errors_health = 0
        self.errors_predict = 0
        self.total_health = 0
        self.total_predict = 0
        self.consecutive_failures_health = 0
        self.consecutive_failures_predict = 0
        self.max_consecutive_failures_health = 0
        self.max_consecutive_failures_predict = 0

    def get_alert_level(self, value, warning, critical):
        if value is None:
            return "grey"
        if value >= critical:
            return "red"
        if value >= warning:
            return "yellow"
        return "green"

    def health(self):
        start = time.time()
        try:
            response = requests.get(f'{self.api_url}/health', timeout=self.config['monitoring']['request_timeout_seconds'])
            latency = time.time() - start
            self.latencies_health.append(latency)
            self.total_health += 1
            if response.ok:
                self.consecutive_failures_health = 0
                self.logger.info(f"[health] status: True, latency: {latency:.3f}, response: {response.json()}")
                self.logger.info(f"metric health_latency {latency:.3f}")
                return True, latency, response.json()
            else:
                self.errors_health += 1
                self.consecutive_failures_health += 1
                if self.consecutive_failures_health > self.max_consecutive_failures_health:
                    self.max_consecutive_failures_health = self.consecutive_failures_health
                self.logger.error(f"[health] status: False, latency: {latency:.3f}, response: {response.text}")
                return False, latency, response.text
        except Exception as e:
            latency = time.time() - start
            self.latencies_health.append(latency)
            self.errors_health += 1
            self.consecutive_failures_health += 1
            if self.consecutive_failures_health > self.max_consecutive_failures_health:
                self.max_consecutive_failures_health = self.consecutive_failures_health
            self.logger.error(f"[health] Exception: {str(e)}")
            return False, latency, str(e)

    def predict(self, img_path):
        def get_type(filename):
            ext = os.path.splitext(filename)[1].lower()
            if ext in [".jpg", ".jpeg"]:
                return "image/jpeg"
            if ext == ".png":
                return "image/png"
            return "application/octet-stream"

        start = time.time()
        try:
            mime_type = get_type(img_path)
            with open(img_path, 'rb') as f:
                files = {'file': (os.path.basename(img_path), f, mime_type)}
                response = requests.post(f'{self.api_url}/predict', files=files, timeout=self.config['monitoring']['request_timeout_seconds'])
            latency = time.time() - start
            self.latencies_predict.append(latency)
            self.total_predict += 1
            if response.ok:
                self.consecutive_failures_predict = 0
                self.logger.info(f"[predict] status: True, latency: {latency:.3f}, response: {response.json()}")
                self.logger.info(f"metric predict_latency {latency:.3f}")
                return True, latency, response.json()
            else:
                self.errors_predict += 1
                self.consecutive_failures_predict += 1
                if self.consecutive_failures_predict > self.max_consecutive_failures_predict:
                    self.max_consecutive_failures_predict = self.consecutive_failures_predict
                self.logger.error(f"[predict] status: False, latency: {latency:.3f}, response: {response.text}")
                return False, latency, response.text
        except Exception as e:
            latency = time.time() - start
            self.latencies_predict.append(latency)
            self.errors_predict += 1
            self.consecutive_failures_predict += 1
            if self.consecutive_failures_predict > self.max_consecutive_failures_predict:
                self.max_consecutive_failures_predict = self.consecutive_failures_predict
            self.logger.error(f"[predict] Exception: {str(e)}")
            return False, latency, str(e)

    def p95_latency(self, latencies):
        if not latencies:
            return None
        sorted_latencies = sorted(latencies)
        idx = int(len(sorted_latencies) * 0.95) - 1
        return sorted_latencies[idx]

    def error_rate(self, errors, total):
        if total == 0:
            return None
        return errors / total

    def run_monitor(self):
        img_files = glob.glob(os.path.join(self.img_folder, '*'))

        for _ in range(self.config['monitoring']['samples_per_check']):
            health_ok, health_latency, health_response = self.health()

            if img_files:
                predict_ok, predict_latency, predict_response = self.predict(img_files[0])
            else:
                self.logger.warning("No images found for prediction testing.")
                predict_ok, predict_latency, predict_response = None, None, None

            p95_health = self.p95_latency(self.latencies_health)
            p95_predict = self.p95_latency(self.latencies_predict)
            error_rate_health = self.error_rate(self.errors_health, self.total_health)
            error_rate_predict = self.error_rate(self.errors_predict, self.total_predict)

            self.logger.info(f"Health P95 latency: {p95_health}")
            self.logger.info(f"Predict P95 latency: {p95_predict}")
            self.logger.info(f"Health error rate: {error_rate_health}")
            self.logger.info(f"Predict error rate: {error_rate_predict}")
            self.logger.info(f"Max consecutive failures health: {self.max_consecutive_failures_health}")
            self.logger.info(f"Max consecutive failures predict: {self.max_consecutive_failures_predict}")
