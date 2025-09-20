#!/usr/bin/env python3
"""
Quick test script to verify DataHub integration works when DataHub is running.
This script simulates the same validation process but with enhanced DataHub testing.
"""

import sys
import requests
from src.validate_data import validate_data


def check_datahub_status():
    """Check if DataHub is running and accessible"""
    try:
        response = requests.get("http://localhost:9002/api/v2/system/config", timeout=5)
        if response.status_code == 200:
            print("✅ DataHub is running and accessible")
            return True
    except requests.RequestException:
        pass

    print("❌ DataHub is not running")
    print("\nTo start DataHub:")
    print("  make datahub-up")
    print("  # Wait about 1-2 minutes for services to start")
    print("  make datahub-check")
    print("\nThen run this test again.")
    return False


def main():
    print("🧪 Testing DataHub Integration")
    print("=" * 40)

    # Check if DataHub is running
    if not check_datahub_status():
        print("\n🤖 Running validation without DataHub (graceful degradation)")

    print("\n📊 Running data validation...")
    try:
        validate_data()
        print("\n✅ Data validation completed successfully!")
        print("📄 Check report: reports/validation/index.html")

        if check_datahub_status():
            print("🏛️  Check DataHub: http://localhost:9002")
            print("   Default login: datahub/datahub")

    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
