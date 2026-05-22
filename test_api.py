
"""
Test script for Healthcare Analytics API
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health Check: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_predict():
    """Test prediction endpoint"""
    data = {
        "age": 45,
        "gender": "Male",
        "blood_type": "O+",
        "medical_condition": "Diabetes",
        "admission_type": "Emergency",
        "billing_amount": 15000.00,
        "length_of_stay": 3,
        "age_group": "Adult",
        "billing_category": "Medium",
        "medication": "Metformin"
    }

    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"\nPrediction: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_model_info():
    """Test model info endpoint"""
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"\nModel Info: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_statistics():
    """Test statistics endpoint"""
    response = requests.get(f"{BASE_URL}/statistics")
    print(f"\nStatistics: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

if __name__ == "__main__":
    print("Testing Healthcare Analytics API...")
    print("="*50)

    try:
        results = []
        results.append(("Health", test_health()))
        results.append(("Predict", test_predict()))
        results.append(("Model Info", test_model_info()))
        results.append(("Statistics", test_statistics()))

        print("\n" + "="*50)
        print("Test Results:")
        for name, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{name}: {status}")

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("Make sure the API is running on http://localhost:8000")
        sys.exit(1)
