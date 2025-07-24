#!/usr/bin/env python3

import requests
import json

# Test the API locally
BASE_URL = "http://localhost:8000"

def test_root():
    print("Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_health():
    print("\nTesting health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_prediction():
    print("\nTesting prediction endpoint...")
    test_data = {
        "age": 25,
        "gender": "Male",
        "region": "South",
        "urban_or_rural": "Urban",
        "household_income_bracket": "Middle",
        "field_of_study": "Engineering",
        "university_type": "Federal",
        "gpa_or_class_of_degree": "Second Class Upper",
        "has_postgrad_degree": "No",
        "years_since_graduation": 2
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=test_data)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Predicted Salary: {result['formatted_salary']}")
            print(f"Confidence: {result['model_confidence']}")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Nigerian Graduate Salary Prediction API")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    if test_root():
        success_count += 1
    
    if test_health():
        success_count += 1
        
    if test_prediction():
        success_count += 1
    
    print(f"\n{'='*50}")
    print(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 All API tests passed! The API is working correctly.")
    else:
        print("❌ Some tests failed. Check the API implementation.")
