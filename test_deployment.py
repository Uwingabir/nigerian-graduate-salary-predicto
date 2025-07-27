#!/usr/bin/env python3
"""
Test script for Render deployment
Replace YOUR_RENDER_URL with your actual Render URL after deployment
"""
import requests
import json

# Replace this with your actual Render URL after deployment
RENDER_URL = "https://your-app-name.onrender.com"  # Update this after deployment

def test_api_endpoints():
    """Test all API endpoints"""
    
    print("🚀 Testing Nigerian Graduate Salary API")
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
    
    # Test 2: Health check endpoint
    print("\n2️⃣ Testing Health Check (GET /health)...")
    try:
        response = requests.get(f"{RENDER_URL}/health", timeout=30)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 200
        print("   ✅ Health check working!")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False
    
    # Test 3: Model info endpoint
    print("\n3️⃣ Testing Model Info (GET /model_info)...")
    try:
        response = requests.get(f"{RENDER_URL}/model_info", timeout=30)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 200
        print("   ✅ Model info working!")
    except Exception as e:
        print(f"   ❌ Model info failed: {e}")
        return False
    
    # Test 4: Prediction endpoint
    print("\n4️⃣ Testing Prediction (POST /predict)...")
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
            "has_postgrad_degree": "No",
            "years_since_graduation": 2
        }
        
        response = requests.post(
            f"{RENDER_URL}/predict", 
            json=test_data, 
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        print(f"   Status: {response.status_code}")
        result = response.json()
        print(f"   Predicted Salary: {result.get('formatted_salary', 'N/A')}")
        print(f"   Confidence: {result.get('model_confidence', 'N/A')}")
        assert response.status_code == 200
        assert 'predicted_salary' in result
        print("   ✅ Prediction working!")
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Your API is working correctly on Render!")
    return True

if __name__ == "__main__":
    print("Before running this test:")
    print("1. Deploy your app to Render")
    print("2. Update the RENDER_URL variable above with your actual URL")
    print("3. Run: pip install requests")
    print("4. Run this script\n")
    
    # Uncomment the next line after updating RENDER_URL
    # test_api_endpoints()
    
    print("Please update the RENDER_URL variable first!")
