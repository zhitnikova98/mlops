# Step 4.5: DataHub Integration with Great Expectations

ML pipeline with Great Expectations integrated with DataHub for metadata management.

## Prerequisites

- Docker and Docker Compose installed
- At least 8GB RAM available for DataHub services

## Setup

1. Install Python dependencies:
```bash
make install
```

2. Start DataHub services:
```bash
make datahub-up
```

3. Initialize DVC:
```bash
dvc init --no-scm || true
dvc remote add -d local ../../.dvcstore || true
```

## Usage

1. Check DataHub status:
```bash
make datahub-check
```

2. Run the ML pipeline:
```bash
make repro
```

3. Access DataHub web interface:
   - URL: http://localhost:9002
   - Default credentials: datahub/datahub

4. Stop DataHub services:
```bash
make datahub-down
```

## Features

- **Data Validation**: Great Expectations validation with DataHub integration
- **Metadata Tracking**: Dataset properties and lineage sent to DataHub
- **Validation Results**: Assertion results tracked in DataHub
- **Web Interface**: Beautiful HTML reports + DataHub dashboard
- **Graceful Degradation**: Pipeline works even if DataHub is unavailable

## DataHub Integration

The pipeline automatically:
- Sends dataset metadata (rows, columns, size) to DataHub
- Creates assertion results for each Great Expectations validation
- Provides links between validation report and DataHub interface
- Tracks data lineage and quality metrics
