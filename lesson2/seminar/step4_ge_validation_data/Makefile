.PHONY: install clean repro demo-fail

install:
	pip install -r requirements.txt

repro:
	dvc repro

demo-fail:
	@echo "🚨 Preparing demo: downloading original data first..."
	dvc repro get_data
	@echo "🚨 Creating bad data that will fail validation..."
	python src/create_bad_data.py
	@echo "📊 Running validation with bad data (should fail)..."
	python src/validate_data.py || echo "✅ Pipeline failed as expected! Validation caught bad data."
	@echo "📋 Check validation report: reports/validation/index.html"

clean:
	rm -rf data/processed models metrics reports
