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
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    model = joblib.load(os.path.join(current_dir, "best_model.pkl"))
    scaler = joblib.load(os.path.join(current_dir, "scaler.pkl"))
    label_encoders = joblib.load(os.path.join(current_dir, "label_encoders.pkl"))
    feature_names = joblib.load(os.path.join(current_dir, "feature_names.pkl"))
    print("Models and preprocessing objects loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    scaler = None
    label_encoders = None
    feature_names = None

# Pydantic model for input validation
class PredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=70, description="Age of the graduate")
    gender: str = Field(..., description="Gender (Male/Female)")
    state_of_origin: str = Field(..., description="State of origin")
    region: str = Field(..., description="Geographical region")
    urban_or_rural: str = Field(..., description="Urban or Rural")
    household_income_bracket: str = Field(..., description="Household income bracket")
    field_of_study: str = Field(..., description="Field of study")
    university_type: str = Field(..., description="University type (Public/Private)")
    gpa_or_class_of_degree: str = Field(..., description="GPA or class of degree")
    has_postgrad_degree: str = Field(..., description="Has postgraduate degree (Yes/No)")
    years_since_graduation: int = Field(..., ge=0, le=50, description="Years since graduation")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🇳🇬 Nigerian Graduate Salary Prediction API",
        "status": "active",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "failed"
    return {
        "status": "healthy",
        "model_status": model_status,
        "api_version": "1.0.0"
    }

@app.post("/predict")
async def predict_salary(request: PredictionRequest):
    """Predict graduate salary"""
    if model is None or scaler is None or label_encoders is None:
        raise HTTPException(status_code=500, detail="Models not loaded properly")
    
    try:
        # Create input dataframe
        input_data = pd.DataFrame({
            'Age': [request.age],
            'Gender': [request.gender],
            'State_of_Origin': [request.state_of_origin],
            'Region': [request.region],
            'Urban_or_Rural': [request.urban_or_rural],
            'Household_Income_Bracket': [request.household_income_bracket],
            'Field_of_Study': [request.field_of_study],
            'University_Type': [request.university_type],
            'GPA_or_Class_of_Degree': [request.gpa_or_class_of_degree],
            'Has_Postgrad_Degree': [request.has_postgrad_degree],
            'Years_Since_Graduation': [request.years_since_graduation],
            'Employment_Status': ['Employed'],  # Default
            'Salary_Level': ['Medium']  # Default
        })
        
        # Encode categorical variables
        for col in input_data.select_dtypes(include=['object']).columns:
            if col in label_encoders:
                try:
                    input_data[col] = label_encoders[col].transform(input_data[col])
                except ValueError:
                    # Handle unseen categories
                    input_data[col] = 0
        
        # Ensure correct feature order
        if feature_names:
            input_data = input_data[feature_names]
        
        # Make prediction
        if 'Linear' in str(type(model)):
            # Use scaler for linear regression
            input_scaled = scaler.transform(input_data)
            log_prediction = model.predict(input_scaled)[0]
        else:
            # Tree-based models don't need scaling
            log_prediction = model.predict(input_data)[0]
        
        # Convert from log scale to actual salary
        predicted_salary = max(0, np.exp(log_prediction))
        
        return {
            "predicted_salary": round(predicted_salary, 2),
            "currency": "NGN",
            "model_type": str(type(model).__name__),
            "input_processed": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# Vercel serverless function handler
def handler(request):
    """Vercel serverless function handler"""
    return app
