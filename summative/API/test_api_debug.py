#!/usr/bin/env python3
import requests
import json

# Test the API health check first
def test_health():
    try:
        response = requests.get("https://nigerian-graduate-salary-predicto-3.onrender.com/health")
        print("Health Check Response:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

# Test model info
def test_model_info():
    try:
        response = requests.get("https://nigerian-graduate-salary-predicto-3.onrender.com/model_info")
        print("\nModel Info Response:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Model info failed: {e}")
        return False

# Test prediction with simple data
def test_prediction():
    try:
        test_data = {
            "age": 25,
            "gender": "Male",
            "region": "South",
            "urban_or_rural": "Urban",
            "household_income_bracket": "Middle",
            "field_of_study": "Engineering",
            "university_type": "Federal",
            "gpa_or_class_of_degree": "Second Class Upper",
            "has_postgrad_degree": "Yes",
            "years_since_graduation": 2
        }
        
        response = requests.post(
            "https://nigerian-graduate-salary-predicto-3.onrender.com/predict",
            headers={"Content-Type": "application/json"},
            json=test_data
        )
        
        print("\nPrediction Response:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Predicted Salary: {result.get('predicted_salary')}")
            print(f"Formatted Salary: {result.get('formatted_salary')}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Prediction test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Nigerian Graduate Salary API ===")
    
    # Test health check
    health_ok = test_health()
    
    # Test model info
    info_ok = test_model_info()
    
    # Test prediction
    pred_ok = test_prediction()
    
    print("\n=== Test Summary ===")
    print(f"Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Model Info: {'✅ PASS' if info_ok else '❌ FAIL'}")
    print(f"Prediction: {'✅ PASS' if pred_ok else '❌ FAIL'}")
