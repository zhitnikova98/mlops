# Step 4: Data Validation with Great Expectations

ML pipeline with data validation using Great Expectations.

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
- Generates HTML validation report
- Stops pipeline if validation fails
