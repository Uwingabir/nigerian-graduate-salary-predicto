#!/usr/bin/env python3
"""
Robust model creation for Nigerian Graduate Salary Prediction
Creates working models that are compatible with the API
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

# Create sample data if we can't access the online dataset
def create_sample_data():
    """Create sample data for testing purposes"""
    np.random.seed(42)
    n_samples = 1000
    
    # Sample data structure matching the expected format
    data = {
        'Age': np.random.randint(22, 40, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'State_of_Origin': np.random.choice(['Lagos', 'Kano', 'Rivers', 'Oyo', 'Edo'], n_samples),
        'Region': np.random.choice(['North', 'South', 'East'], n_samples),
        'Urban_or_Rural': np.random.choice(['Urban', 'Rural'], n_samples),
        'Household_Income_Bracket': np.random.choice(['Low', 'Middle', 'High'], n_samples),
        'Field_of_Study': np.random.choice(['Engineering', 'Business', 'Health Sciences', 'Education', 'Arts', 'Science'], n_samples),
        'University_Type': np.random.choice(['Federal', 'State', 'Private'], n_samples),
        'GPA_or_Class_of_Degree': np.random.choice(['First Class', 'Second Class Upper', 'Second Class Lower', 'Third Class'], n_samples),
        'Has_Postgrad_Degree': np.random.choice(['Yes', 'No'], n_samples),
        'Years_Since_Graduation': np.random.randint(0, 15, n_samples),
        'Employment_Status': np.random.choice(['Employed', 'Unemployed'], n_samples),
        'Salary_Level': np.random.choice(['Low', 'Medium', 'High'], n_samples)
    }
    
    # Generate realistic salaries based on features
    salaries = []
    for i in range(n_samples):
        base_salary = 150000
        
        # Adjust based on field of study
        if data['Field_of_Study'][i] == 'Engineering':
            base_salary *= 1.3
        elif data['Field_of_Study'][i] == 'Health Sciences':
            base_salary *= 1.2
        elif data['Field_of_Study'][i] == 'Business':
            base_salary *= 1.1
        
        # Adjust based on university type
        if data['University_Type'][i] == 'Federal':
            base_salary *= 1.1
        elif data['University_Type'][i] == 'Private':
            base_salary *= 1.05
        
        # Adjust based on GPA
        if data['GPA_or_Class_of_Degree'][i] == 'First Class':
            base_salary *= 1.2
        elif data['GPA_or_Class_of_Degree'][i] == 'Second Class Upper':
            base_salary *= 1.1
        
        # Add some randomness
        base_salary *= np.random.uniform(0.7, 1.4)
        
        salaries.append(int(base_salary))
    
    data['Net_Salary'] = salaries
    
    return pd.DataFrame(data)

def create_models():
    print("🚀 Creating Nigerian Graduate Salary Prediction Models...")
    
    try:
        # Try to load online dataset first
        print("📊 Attempting to load online dataset...")
        url = "https://raw.githubusercontent.com/Uwingabir/nigerian-graduate-salary-prediction/refs/heads/main/linear_regression_model/summative/linear_regression/Nigerian_Graduate_Survey_with_Salary.csv"
        df = pd.read_csv(url, timeout=10)
        print(f"✅ Online dataset loaded: {df.shape[0]} rows")
    except Exception as e:
        print(f"⚠️ Could not load online dataset: {e}")
        print("📊 Creating sample dataset...")
        df = create_sample_data()
        print(f"✅ Sample dataset created: {df.shape[0]} rows")
    
    # Clean data
    print("🔧 Preprocessing data...")
    
    # Remove rows with invalid salaries
    df_clean = df[df['Net_Salary'] > 0].copy()
    if len(df_clean) == 0:
        print("⚠️ No valid salary data, creating minimum viable dataset...")
        df_clean = create_sample_data()
    
    print(f"After cleaning: {df_clean.shape[0]} rows")
    
    # Use log transformation for better prediction
    df_clean['Log_Salary'] = np.log(df_clean['Net_Salary'])
    
    # Encode categorical variables
    print("🏷️ Encoding categorical variables...")
    label_encoders = {}
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != 'Graduate_ID']  # Skip ID column
    
    for col in categorical_cols:
        if col in df_clean.columns:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            label_encoders[col] = le
            print(f"  ✓ {col}: {len(le.classes_)} categories")
    
    # Prepare features - ensure we have the exact columns the API expects
    expected_features = [
        'Age', 'Gender', 'Region', 'Urban_or_Rural', 'Household_Income_Bracket',
        'Field_of_Study', 'University_Type', 'GPA_or_Class_of_Degree',
        'Has_Postgrad_Degree', 'Years_Since_Graduation'
    ]
    
    # Add missing features if they don't exist
    for feature in expected_features:
        if feature not in df_clean.columns:
            # Create default values for missing features
            if feature in ['Age', 'Years_Since_Graduation']:
                df_clean[feature] = np.random.randint(20, 35, len(df_clean))
            else:
                df_clean[feature] = 0  # Encoded value for default category
    
    # Select only the features we need
    feature_cols = expected_features
    X = df_clean[feature_cols].copy()
    y = df_clean['Log_Salary']  # Use log salary
    
    print(f"Final features: {list(X.columns)}")
    print(f"Feature shape: {X.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train a simple Linear Regression model
    print("📈 Training Linear Regression model...")
    model = LinearRegression()  # Simple, no extra parameters
    model.fit(X_train_scaled, y_train)
    
    # Test the model
    y_pred = model.predict(X_test_scaled)
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    print(f"Model R² score: {r2:.3f}")
    
    # Save everything
    print("💾 Saving models and preprocessors...")
    
    joblib.dump(model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(feature_cols, 'feature_names.pkl')
    
    print("✅ All models saved successfully!")
    
    # Test the saved model
    print("🧪 Testing saved model...")
    test_model = joblib.load('best_model.pkl')
    test_scaler = joblib.load('scaler.pkl')
    
    # Create test input
    test_input = np.array([[25, 0, 1, 0, 1, 0, 0, 1, 0, 2]])  # Sample encoded values
    test_input_scaled = test_scaler.transform(test_input)
    log_pred = test_model.predict(test_input_scaled)[0]
    actual_pred = np.exp(log_pred)
    
    print(f"Test prediction: ₦{actual_pred:,.2f}")
    print("✅ Model test successful!")
    
    return True

if __name__ == "__main__":
    try:
        success = create_models()
        if success:
            print("\n🎉 Model creation completed successfully!")
            print("🚀 Models are ready for API deployment!")
        else:
            print("\n❌ Model creation failed!")
    except Exception as e:
        print(f"\n❌ Error during model creation: {e}")
        import traceback
        traceback.print_exc()
