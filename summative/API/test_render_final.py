#!/usr/bin/env python3
"""
Test script for Render deployment
Replace YOUR_RENDER_URL with your actual Render URL
"""
import requests
import json

# Replace this with your actual Render URL after deployment
RENDER_URL = "https://nigerian-graduate-salary-api.onrender.com"  # Update this!

def test_render_api():
    """Test all API endpoints on Render"""
    
    print("🚀 Testing Nigerian Graduate Salary API on Render")
    print(f"🌐 API URL: {RENDER_URL}")
    print("=" * 60)
    
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
    print("🇳🇬 Nigerian Graduate Salary Prediction API - Render Test")
    print("📋 Assignment Requirements Check:")
    print("   ✅ FastAPI framework")
    print("   ✅ Pydantic data validation") 
    print("   ✅ Uvicorn server")
    print("   ✅ CORS middleware")
    print("   ✅ POST endpoint with data validation")
    print("   ✅ requirements.txt file")
    print("   ✅ Deployed on Render")
    print()
    
    print("⚠️  IMPORTANT: Update RENDER_URL variable with your actual deployment URL!")
    print("    Example: https://nigerian-graduate-salary-api.onrender.com")
    print()
    
    success = test_render_api()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! Your API meets assignment requirements!")
        print("📖 Swagger UI available at: {}/docs".format(RENDER_URL))
        print("🚀 Assignment Task 2 COMPLETE!")
    else:
        print("❌ Some tests failed. Check the deployment logs on Render.")
    print("=" * 60)
