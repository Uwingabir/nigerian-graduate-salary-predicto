from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from typing import Union

app = FastAPI(
    title="Nigerian Graduate Salary Prediction API",
    description="Predict graduate salaries based on education, demographics, and socioeconomic factors",
    version="2.0.0"  # Updated version to force refresh
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class GraduateInput(BaseModel):
    age: int = Field(..., ge=20, le=50, description="Age of the graduate (20-50 years)")
    gender: str = Field(..., pattern="^(Male|Female)$", description="Gender: Male or Female")
    region: str = Field(..., pattern="^(North|South|East)$", description="Region: North, South, or East")
    urban_or_rural: str = Field(..., pattern="^(Urban|Rural)$", description="Area type: Urban or Rural")
    household_income_bracket: str = Field(..., pattern="^(Low|Middle|High)$", description="Household income: Low, Middle, or High")
    field_of_study: str = Field(..., pattern="^(Engineering|Business|Health Sciences|Education|Arts|Science)$", 
                               description="Field of study: Engineering, Business, Health Sciences, Education, Arts, or Science")
    university_type: str = Field(..., pattern="^(Federal|State|Private)$", description="University type: Federal, State, or Private")
    gpa_or_class_of_degree: str = Field(..., pattern="^(First Class|Second Class Upper|Second Class Lower|Third Class)$", 
                                       description="GPA/Class: First Class, Second Class Upper, Second Class Lower, or Third Class")
    has_postgrad_degree: str = Field(..., pattern="^(Yes|No)$", description="Has postgraduate degree: Yes or No")
    years_since_graduation: int = Field(..., ge=0, le=20, description="Years since graduation (0-20 years)")

class PredictionResponse(BaseModel):
    predicted_salary: float
    formatted_salary: str
    input_data: dict
    model_confidence: str

@app.get("/")
async def root():
    return {
        "message": "Nigerian Graduate Salary Prediction API",
        "status": "operational",
        "version": "1.0.0",
        "endpoints": {
            "/docs": "API documentation",
            "/predict": "POST endpoint for salary prediction"
        }
    }

@app.get("/health")
async def health_check():
    import os
    pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
    return {
        "status": "healthy",
        "model_status": "rule-based predictor active",
        "api_version": "2.0.0",
        "pkl_files_found": pkl_files,
        "working_directory": os.getcwd(),
        "python_path": os.environ.get('PYTHONPATH', 'Not set')
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_salary(graduate_data: GraduateInput):
    """
    Predict graduate salary based on input features using a simplified rule-based model
    """
    print("🔄 Prediction request received - Using rule-based model (no ML files)")
    try:
        # Convert input to dictionary
        input_dict = graduate_data.dict()
        
        # Simple rule-based prediction (no ML models to avoid errors)
        base_salary = 150000  # Base salary in Naira
        
        # Adjust based on factors
        salary_multiplier = 1.0
        
        # Age factor
        if input_dict['age'] > 30:
            salary_multiplier += 0.2
        elif input_dict['age'] > 25:
            salary_multiplier += 0.1
        
        # Field of study factor
        field_multipliers = {
            'Engineering': 1.3,
            'Health Sciences': 1.2,
            'Business': 1.1,
            'Science': 1.0,
            'Education': 0.9,
            'Arts': 0.8
        }
        salary_multiplier *= field_multipliers.get(input_dict['field_of_study'], 1.0)
        
        # University type factor
        university_multipliers = {
            'Federal': 1.2,
            'State': 1.0,
            'Private': 1.1
        }
        salary_multiplier *= university_multipliers.get(input_dict['university_type'], 1.0)
        
        # GPA factor
        gpa_multipliers = {
            'First Class': 1.3,
            'Second Class Upper': 1.1,
            'Second Class Lower': 0.95,
            'Third Class': 0.8
        }
        salary_multiplier *= gpa_multipliers.get(input_dict['gpa_or_class_of_degree'], 1.0)
        
        # Postgraduate degree factor
        if input_dict['has_postgrad_degree'] == 'Yes':
            salary_multiplier += 0.2
        
        # Experience factor
        salary_multiplier += (input_dict['years_since_graduation'] * 0.05)
        
        # Urban/Rural factor
        if input_dict['urban_or_rural'] == 'Urban':
            salary_multiplier += 0.1
        
        # Income bracket factor
        income_multipliers = {
            'High': 1.2,
            'Middle': 1.0,
            'Low': 0.9
        }
        salary_multiplier *= income_multipliers.get(input_dict['household_income_bracket'], 1.0)
        
        # Calculate final prediction
        prediction = base_salary * salary_multiplier
        
        # Add some randomness to make it more realistic
        import random
        random.seed(hash(str(input_dict)) % 2147483647)  # Deterministic randomness
        variation = random.uniform(0.85, 1.15)
        prediction *= variation
        
        # Ensure reasonable bounds
        prediction = max(50000, min(500000, prediction))
        
        # Format the salary
        formatted_salary = f"₦{prediction:,.2f}"
        
        # Determine confidence level based on prediction range
        if prediction < 100000:
            confidence = "Low-paying job range"
        elif prediction < 300000:
            confidence = "Moderate salary range"
        else:
            confidence = "Well-paying job range"
        
        return PredictionResponse(
            predicted_salary=float(prediction),
            formatted_salary=formatted_salary,
            input_data=input_dict,
            model_confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """
    Get information about the model and features
    """
    return {
        "model_type": "Nigerian Graduate Salary Prediction (Rule-based)",
        "features": [
            "Age (20-50 years)",
            "Gender (Male/Female)",
            "Region (North/South/East)",
            "Urban or Rural area",
            "Household Income Bracket (Low/Middle/High)",
            "Field of Study (Engineering/Business/Health Sciences/Education/Arts/Science)",
            "University Type (Federal/State/Private)",
            "GPA or Class of Degree (First Class/Second Class Upper/Second Class Lower/Third Class)",
            "Has Postgraduate Degree (Yes/No)",
            "Years Since Graduation (0-20 years)"
        ],
        "target": "Net Salary in Nigerian Naira (₦)",
        "purpose": "Help young Africans make informed career and education decisions",
        "model_version": "Rule-based v1.0 (Simplified for reliability)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
