from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from typing import Union
import os

app = FastAPI(
    title="Nigerian Graduate Salary Prediction API",
    description="Predict graduate salaries based on education, demographics, and socioeconomic factors",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load models and preprocessing objects on startup
try:
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    feature_names = joblib.load("feature_names.pkl")
    print("Models and preprocessing objects loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    model = scaler = label_encoders = feature_names = None

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
        "description": "Predict graduate salaries based on education, demographics, and socioeconomic factors",
        "endpoints": {
            "/docs": "API documentation",
            "/predict": "POST endpoint for salary prediction"
        }
    }

@app.get("/health")
async def health_check():
    model_status = "loaded" if model is not None else "not loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "api_version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_salary(graduate_data: GraduateInput):
    """
    Predict graduate salary based on input features
    """
    if model is None or scaler is None or label_encoders is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")
    
    try:
        # Convert input to dictionary
        input_dict = graduate_data.dict()
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'Age': [input_dict['age']],
            'Gender': [input_dict['gender']],
            'Region': [input_dict['region']],
            'Urban_or_Rural': [input_dict['urban_or_rural']],
            'Household_Income_Bracket': [input_dict['household_income_bracket']],
            'Field_of_Study': [input_dict['field_of_study']],
            'University_Type': [input_dict['university_type']],
            'GPA_or_Class_of_Degree': [input_dict['gpa_or_class_of_degree']],
            'Has_Postgrad_Degree': [input_dict['has_postgrad_degree']],
            'Years_Since_Graduation': [input_dict['years_since_graduation']]
        })
        
        # Encode categorical variables
        categorical_cols = ['Gender', 'Region', 'Urban_or_Rural', 'Household_Income_Bracket', 
                           'Field_of_Study', 'University_Type', 'GPA_or_Class_of_Degree', 'Has_Postgrad_Degree']
        
        for col in categorical_cols:
            if col in label_encoders:
                try:
                    input_data[col] = label_encoders[col].transform(input_data[col])
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=f"Invalid value for {col}: {input_data[col].iloc[0]}")
        
        # Make prediction
        # Assuming the best model requires scaling (adjust based on your actual best model)
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        # Ensure non-negative salary
        prediction = max(0, prediction)
        
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
        "model_type": "Nigerian Graduate Salary Prediction",
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
        "purpose": "Help young Africans make informed career and education decisions"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
