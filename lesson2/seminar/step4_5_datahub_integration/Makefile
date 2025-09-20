.PHONY: install clean repro datahub-up datahub-down datahub-check

install:
	pip install -r requirements.txt

repro: install
	dvc init --no-scm || true
	dvc remote add -d local ../../.dvcstore || true
	dvc repro

datahub-up:
	docker-compose up -d
	@echo "Waiting for DataHub to start..."
	@sleep 30
	@echo "DataHub should be available at http://localhost:9002"

datahub-down:
	docker-compose down

datahub-check:
	@curl -f http://localhost:9002/api/v2/system/config > /dev/null 2>&1 && echo "DataHub is running" || echo "DataHub is not available"

clean:
	rm -rf data/ processed/ models/ metrics/ reports/ .dvc/ .dvcignore .venv/ .pytest_cache/ .mypy_cache/ gx/ datahub_data/
