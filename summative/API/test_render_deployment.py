#!/usr/bin/env python3
"""
Test script for the deployed Render API
"""
import requests
import json

# You'll need to replace this with your actual Render URL
RENDER_URL = "https://nigerian-graduate-salary-api.onrender.com"

def test_api_endpoints():
    """Test all API endpoints"""
    
    print("🚀 Testing Render Deployment...")
    print(f"🌐 API URL: {RENDER_URL}")
    print("=" * 50)
    
    # Test 1: Root endpoint
    print("1️⃣ Testing Root Endpoint (GET /)...")
    try:
        response = requests.get(f"{RENDER_URL}/", timeout=30)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 200
        print("   ✅ Root endpoint working!")
    except Exception as e:
        print(f"   ❌ Root endpoint failed: {e}")
        return False
    
    # Test 2: Health endpoint
    print("\n2️⃣ Testing Health Endpoint (GET /health)...")
    try:
        response = requests.get(f"{RENDER_URL}/health", timeout=30)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 200
        print("   ✅ Health endpoint working!")
    except Exception as e:
        print(f"   ❌ Health endpoint failed: {e}")
        return False
    
    # Test 3: Prediction endpoint
    print("\n3️⃣ Testing Prediction Endpoint (POST /predict)...")
    test_data = {
        "age": 25,
        "gender": "Female",
        "state_of_origin": "Lagos",
        "region": "South West",
        "urban_or_rural": "Urban",
        "household_income_bracket": "Middle",
        "field_of_study": "Engineering",
        "university_type": "Public",
        "gpa_or_class_of_degree": "Second Upper",
        "has_postgrad_degree": "No",
        "years_since_graduation": 2
    }
    
    try:
        response = requests.post(
            f"{RENDER_URL}/predict", 
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        if response.status_code == 200:
            result = response.json()
            predicted_salary = result.get('predicted_salary', 0)
            print(f"   💰 Predicted Salary: ₦{predicted_salary:,.0f}")
            print("   ✅ Prediction endpoint working!")
            return True
        else:
            print(f"   ❌ Prediction failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Prediction endpoint failed: {e}")
        return False

if __name__ == "__main__":
    print("🇳🇬 Nigerian Graduate Salary Prediction API Test")
    print("🔗 Testing Render Deployment")
    print()
    
    # Note: Replace the URL below with your actual Render URL
    print("⚠️  IMPORTANT: Update RENDER_URL variable with your actual deployment URL!")
    print("    Example: https://your-service-name.onrender.com")
    print()
    
    success = test_api_endpoints()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED! Your API is working on Render!")
        print("🚀 Your API is ready for production use!")
    else:
        print("❌ Some tests failed. Check the deployment logs.")
    print("=" * 50)
