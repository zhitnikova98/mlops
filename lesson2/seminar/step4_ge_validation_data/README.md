# Step 4: Data Validation with Great Expectations

ML pipeline with data validation using Great Expectations and Data Docs.

## Setup
```bash
make install
dvc init
dvc remote add -d local ../../.dvcstore
```

## Run
```bash
dvc repro
```

Pipeline includes data validation step that:
- Checks for null values in total_bill, tip, size
- Validates total_bill range [0, 100]
- Validates size range [1, 10]
- Generates beautiful HTML validation report at `reports/validation/index.html`
- Stops pipeline if validation fails

Open `reports/validation/index.html` in browser to see detailed validation report with styled results and statistics.

## Fail Fast Demo

Demonstrate how Great Expectations catches bad data and stops the pipeline:

```bash
make demo-fail
```

This will:
1. Create dataset with validation violations:
   - `total_bill`: contains null values and value > 100
   - `tip`: contains null values
   - `size`: contains values < 1 and > 10
2. Run DVC pipeline
3. **Pipeline fails at validation step** ❌
4. Generate validation report showing specific violations
5. **No further stages execute** (no preprocessing, training, evaluation)

This demonstrates **fail fast** principle: catch data quality issues early, before expensive ML training.
