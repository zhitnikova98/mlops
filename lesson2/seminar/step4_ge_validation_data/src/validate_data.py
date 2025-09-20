import os
import sys
import pandas as pd
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
from great_expectations.dataset.pandas_dataset import PandasDataset


def validate_data():
    df = pd.read_csv("data/raw/tips.csv")

    suite = ExpectationSuite(expectation_suite_name="tips_validation")

    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "total_bill"},
        )
    )

    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "tip"},
        )
    )

    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "size"},
        )
    )

    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "total_bill", "min_value": 0, "max_value": 100},
        )
    )

    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "size", "min_value": 1, "max_value": 10},
        )
    )

    ge_df = PandasDataset(df)
    results = ge_df.validate(expectation_suite=suite)

    os.makedirs("reports/validation", exist_ok=True)

    success_rate = (
        results.statistics["successful_expectations"]
        / results.statistics["evaluated_expectations"]
    )

    html_report = f"""
    <html>
    <head><title>Data Validation Report</title></head>
    <body>
        <h1>Tips Dataset Validation Report</h1>
        <p><strong>Success Rate:</strong> {success_rate:.2%}</p>
        <p><strong>Total Expectations:</strong> {results.statistics['evaluated_expectations']}</p>
        <p><strong>Successful:</strong> {results.statistics['successful_expectations']}</p>
        <p><strong>Failed:</strong> {results.statistics['unsuccessful_expectations']}</p>
        <h2>Results:</h2>
        <ul>
    """

    for result in results.results:
        status = "✓" if result.success else "✗"
        html_report += f"<li>{status} {result.expectation_config.expectation_type}</li>"

    html_report += """
        </ul>
    </body>
    </html>
    """

    with open("reports/validation/index.html", "w") as f:
        f.write(html_report)

    if not results.success:
        print("Data validation failed!")
        sys.exit(1)

    print("Data validation passed!")


if __name__ == "__main__":
    validate_data()
