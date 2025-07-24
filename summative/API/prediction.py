import joblib
import pandas as pd
import numpy as np

def load_prediction_models():
    """
    Load all necessary models and preprocessing objects
    """
    try:
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")
        label_encoders = joblib.load("label_encoders.pkl")
        feature_names = joblib.load("feature_names.pkl")
        return model, scaler, label_encoders, feature_names
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None, None

def predict_graduate_salary(age, gender, region, urban_or_rural, household_income_bracket, 
                           field_of_study, university_type, gpa_or_class_of_degree, 
                           has_postgrad_degree, years_since_graduation):
    """
    Predict salary based on input features
    
    Parameters:
    - age: int (20-50)
    - gender: str ('Male' or 'Female')
    - region: str ('North', 'South', or 'East')
    - urban_or_rural: str ('Urban' or 'Rural')
    - household_income_bracket: str ('Low', 'Middle', or 'High')
    - field_of_study: str ('Engineering', 'Business', 'Health Sciences', 'Education', 'Arts', 'Science')
    - university_type: str ('Federal', 'State', or 'Private')
    - gpa_or_class_of_degree: str ('First Class', 'Second Class Upper', 'Second Class Lower', 'Third Class')
    - has_postgrad_degree: str ('Yes' or 'No')
    - years_since_graduation: int (0-20)
    
    Returns:
    - predicted_salary: float
    """
    
    # Load models and preprocessing objects
    model, scaler, label_encoders, feature_names = load_prediction_models()
    
    if model is None:
        raise Exception("Could not load prediction models")
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Region': [region],
        'Urban_or_Rural': [urban_or_rural],
        'Household_Income_Bracket': [household_income_bracket],
        'Field_of_Study': [field_of_study],
        'University_Type': [university_type],
        'GPA_or_Class_of_Degree': [gpa_or_class_of_degree],
        'Has_Postgrad_Degree': [has_postgrad_degree],
        'Years_Since_Graduation': [years_since_graduation]
    })
    
    # Encode categorical variables
    categorical_cols = ['Gender', 'Region', 'Urban_or_Rural', 'Household_Income_Bracket', 
                       'Field_of_Study', 'University_Type', 'GPA_or_Class_of_Degree', 'Has_Postgrad_Degree']
    
    for col in categorical_cols:
        if col in label_encoders:
            input_data[col] = label_encoders[col].transform(input_data[col])
    
    # Scale features (assuming the best model requires scaling)
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Ensure non-negative salary
    return max(0, prediction)

def get_model_info():
    """
    Get information about the available model features and ranges
    """
    return {
        "input_features": {
            "age": {"type": "int", "range": "20-50", "description": "Age of the graduate"},
            "gender": {"type": "str", "options": ["Male", "Female"], "description": "Gender"},
            "region": {"type": "str", "options": ["North", "South", "East"], "description": "Region in Nigeria"},
            "urban_or_rural": {"type": "str", "options": ["Urban", "Rural"], "description": "Area type"},
            "household_income_bracket": {"type": "str", "options": ["Low", "Middle", "High"], "description": "Family income level"},
            "field_of_study": {"type": "str", "options": ["Engineering", "Business", "Health Sciences", "Education", "Arts", "Science"], "description": "Academic field"},
            "university_type": {"type": "str", "options": ["Federal", "State", "Private"], "description": "Type of university"},
            "gpa_or_class_of_degree": {"type": "str", "options": ["First Class", "Second Class Upper", "Second Class Lower", "Third Class"], "description": "Academic performance"},
            "has_postgrad_degree": {"type": "str", "options": ["Yes", "No"], "description": "Postgraduate education"},
            "years_since_graduation": {"type": "int", "range": "0-20", "description": "Experience years"}
        },
        "output": {
            "predicted_salary": {"type": "float", "unit": "Nigerian Naira (₦)", "description": "Predicted annual salary"}
        }
    }

# Example usage
if __name__ == "__main__":
    # Test the prediction function
    try:
        test_prediction = predict_graduate_salary(
            age=25,
            gender='Male',
            region='South',
            urban_or_rural='Urban',
            household_income_bracket='Middle',
            field_of_study='Engineering',
            university_type='Federal',
            gpa_or_class_of_degree='Second Class Upper',
            has_postgrad_degree='Yes',
            years_since_graduation=2
        )
        
        print(f"Test prediction: ₦{test_prediction:,.2f}")
        print("Prediction function is working correctly!")
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        print("Make sure the model files are in the same directory:")
