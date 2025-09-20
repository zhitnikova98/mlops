import os
import sys
import pandas as pd
import yaml
import time
from great_expectations.dataset import PandasDataset
from datahub.emitter.mcp import MetadataChangeProposalWrapper
from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.metadata.schema_classes import (
    DatasetPropertiesClass,
)
from datahub.metadata.com.linkedin.pegasus2avro.assertion import (
    AssertionRunEventClass,
    AssertionResult,
    AssertionType,
)
import requests


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def wait_for_datahub(server_url, timeout=60):
    """Wait for DataHub to be ready"""
    print(f"Waiting for DataHub at {server_url}...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{server_url}/api/v2/system/config", timeout=5)
            if response.status_code == 200:
                print("DataHub is ready!")
                return True
        except requests.RequestException:
            pass
        time.sleep(2)

    print(f"DataHub is not available after {timeout} seconds")
    return False


def send_dataset_metadata_to_datahub(df, emitter):
    """Send dataset metadata to DataHub"""
    dataset_urn = "urn:li:dataset:(urn:li:dataPlatform:file,tips.csv,PROD)"

    # Dataset properties
    dataset_props = DatasetPropertiesClass(
        description="Restaurant tips dataset for ML experiments",
        customProperties={
            "source": "seaborn-data repository",
            "format": "CSV",
            "encoding": "utf-8",
            "rows": str(len(df)),
            "columns": str(len(df.columns)),
            "size_mb": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f}",
        },
    )

    metadata_change_proposal = MetadataChangeProposalWrapper(
        entityType="dataset",
        entityUrn=dataset_urn,
        aspectName="datasetProperties",
        aspect=dataset_props,
    )

    emitter.emit_mcp(metadata_change_proposal)
    print(f"Sent dataset metadata to DataHub: {dataset_urn}")


def send_validation_results_to_datahub(validation_results, emitter):
    """Send Great Expectations validation results to DataHub"""
    for i, result in enumerate(validation_results.results):
        assertion_urn = f"urn:li:assertion:(urn:li:dataPlatform:great-expectations,tips.expectation.{i},PROD)"

        # Create assertion result
        assertion_result = AssertionResult(
            type=AssertionType.DATASET,
            success=result.success,
            url=None,
            externalUrl=None,
            error=None if result.success else "Validation failed",
            actualAggValue=result.result.get("observed_value", 0),
            nativeResults={
                "expectation_type": result.expectation_config.expectation_type,
                "column": result.expectation_config.kwargs.get("column", ""),
                "success": str(result.success),
            },
        )

        assertion_run_event = AssertionRunEventClass(
            timestampMillis=int(time.time() * 1000),
            partitionSpec=None,
            result=assertion_result,
            runId=f"ge-validation-{int(time.time())}",
        )

        metadata_change_proposal = MetadataChangeProposalWrapper(
            entityType="assertion",
            entityUrn=assertion_urn,
            aspectName="assertionRunEvent",
            aspect=assertion_run_event,
        )

        emitter.emit_mcp(metadata_change_proposal)

    print(f"Sent {len(validation_results.results)} validation results to DataHub")


def validate_data():
    params = load_params()
    datahub_server = params["datahub"]["server"]

    # Check if DataHub is available
    if not wait_for_datahub(datahub_server, params["datahub"]["timeout"]):
        print(
            "Warning: DataHub is not available, proceeding without DataHub integration"
        )
        datahub_available = False
    else:
        datahub_available = True

    # Read data
    df = pd.read_csv("data/raw/tips.csv")

    # Initialize DataHub emitter if available
    emitter = None
    if datahub_available:
        try:
            emitter = DatahubRestEmitter(gms_server=datahub_server)
            send_dataset_metadata_to_datahub(df, emitter)
        except Exception as e:
            print(f"Warning: Failed to connect to DataHub: {e}")
            datahub_available = False

    # Create GX DataSet
    ge_df = PandasDataset(df)

    # Create expectations
    ge_df.expect_column_values_to_not_be_null("total_bill")
    ge_df.expect_column_values_to_not_be_null("tip")
    ge_df.expect_column_values_to_not_be_null("size")
    ge_df.expect_column_values_to_be_between("total_bill", min_value=0, max_value=100)
    ge_df.expect_column_values_to_be_between("size", min_value=1, max_value=10)

    # Validate data
    validation_results = ge_df.validate()

    # Send results to DataHub if available
    if datahub_available and emitter:
        try:
            send_validation_results_to_datahub(validation_results, emitter)
        except Exception as e:
            print(f"Warning: Failed to send validation results to DataHub: {e}")

    # Create directories for reports
    os.makedirs("reports/validation", exist_ok=True)

    # Generate HTML report with DataHub integration status
    success_rate = (
        validation_results.statistics["successful_expectations"]
        / validation_results.statistics["evaluated_expectations"]
    )

    datahub_status = "✅ Connected" if datahub_available else "❌ Not Available"
    datahub_url = (
        f'<a href="{datahub_server}" target="_blank">{datahub_server}</a>'
        if datahub_available
        else datahub_server
    )

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Tips Dataset Validation Report with DataHub</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 8px; }}
        .success {{ color: #2e7d32; }}
        .failed {{ color: #d32f2f; }}
        .warning {{ color: #f57c00; }}
        .summary {{ margin: 20px 0; }}
        .datahub-section {{ margin: 20px 0; padding: 15px; background: #e8f5e8; border-radius: 8px; }}
        .expectation {{ margin: 10px 0; padding: 10px; border-left: 4px solid #2196f3; }}
        .expectation.success {{ border-color: #4caf50; background: #f1f8e9; }}
        .expectation.failed {{ border-color: #f44336; background: #ffebee; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Tips Dataset Validation Report with DataHub Integration</h1>
        <div class="summary">
            <p><strong>Success Rate:</strong> <span class="{'success' if success_rate == 1.0 else 'failed'}">{success_rate:.1%}</span></p>
            <p><strong>Total Expectations:</strong> {validation_results.statistics["evaluated_expectations"]}</p>
            <p><strong>Successful:</strong> <span class="success">{validation_results.statistics["successful_expectations"]}</span></p>
            <p><strong>Failed:</strong> <span class="failed">{validation_results.statistics["unsuccessful_expectations"]}</span></p>
        </div>
    </div>

    <div class="datahub-section">
        <h2>🏛️ DataHub Integration</h2>
        <p><strong>Status:</strong> {datahub_status}</p>
        <p><strong>Server:</strong> {datahub_url}</p>
        <p><strong>Metadata Sent:</strong> {'Yes' if datahub_available else 'No'}</p>
        <p><strong>Validation Results Sent:</strong> {'Yes' if datahub_available else 'No'}</p>
    </div>

    <h2>Expectation Results:</h2>
"""

    for result in validation_results.results:
        status = "success" if result.success else "failed"
        status_icon = "✅" if result.success else "❌"
        expectation_type = result.expectation_config.expectation_type
        column = result.expectation_config.kwargs.get("column", "")

        html_content += f"""
    <div class="expectation {status}">
        <strong>{status_icon} {expectation_type}</strong>
        {f'<br><em>Column:</em> {column}' if column else ''}
        <br><em>Success:</em> {result.success}
    </div>"""

    html_content += """
    <div style="margin-top: 40px; padding: 20px; background: #e3f2fd; border-radius: 8px;">
        <h3>Dataset Summary</h3>
        <p><strong>Shape:</strong> {rows} rows × {cols} columns</p>
        <p><strong>Generated:</strong> {timestamp}</p>
        <p><strong>DataHub Integration:</strong> {datahub_status}</p>
    </div>
</body>
</html>""".format(
        rows=len(df),
        cols=len(df.columns),
        timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        datahub_status="Enabled" if datahub_available else "Disabled",
    )

    # Save HTML report
    with open("reports/validation/index.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print("Validation report generated at: reports/validation/index.html")
    if datahub_available:
        print(f"DataHub interface available at: {datahub_server}")

    # Check validation results
    if not validation_results.success:
        print("Data validation failed!")
        sys.exit(1)

    print("Data validation passed!")


if __name__ == "__main__":
    validate_data()
