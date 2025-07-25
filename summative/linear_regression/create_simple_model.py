#!/usr/bin/env python3
"""
Create a simple, working model for the Nigerian Graduate Salary Prediction API
This script creates models that are guaranteed to work without the 'positive' attribute error
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def create_working_models():
    print("🚀 Creating simple, working models...")
    
    # Load the dataset
    print("📊 Loading dataset...")
    url = "https://raw.githubusercontent.com/Uwingabir/nigerian-graduate-salary-prediction/refs/heads/main/linear_regression_model/summative/linear_regression/Nigerian_Graduate_Survey_with_Salary.csv"
    df = pd.read_csv(url)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Simple preprocessing
    print("🔧 Preprocessing data...")
    
    # Drop unnecessary columns and clean data
    df_clean = df.drop('Graduate_ID', axis=1).copy()
    
    # Remove rows with zero or negative salaries
    df_clean = df_clean[df_clean['Net_Salary'] > 0].copy()
    print(f"After cleaning: {df_clean.shape[0]} rows remaining")
    
    # Log transform the salary (helps with prediction accuracy)
    df_clean['Log_Salary'] = np.log(df_clean['Net_Salary'])
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    print("🏷️ Encoding categorical variables:")
    for col in categorical_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
        label_encoders[col] = le
        print(f"  ✓ {col}: {len(le.classes_)} categories")
    
    # Prepare features and target
    X = df_clean.drop(['Net_Salary', 'Log_Salary'], axis=1)
    y = df_clean['Log_Salary']  # Use log salary for better predictions
    
    print(f"Features: {list(X.columns)}")
    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardization (important for Linear Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("📈 Training models...")
    
    # Create simple models WITHOUT any problematic parameters
    models = {
        'Linear Regression': LinearRegression(),  # Basic LinearRegression, no extra params
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"  🔄 Training {name}...")
        
        try:
            # Linear Regression needs scaled data, tree models don't
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Evaluate on log scale
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Convert back to actual salary for real-world interpretation
            y_test_actual = np.exp(y_test)
            y_pred_actual = np.exp(y_pred)
            mae_actual = np.mean(np.abs(y_test_actual - y_pred_actual))
            
            results[name] = {
                'R²': r2,
                'MSE': mse,
                'MAE_Actual': mae_actual
            }
            trained_models[name] = model
            
            print(f"    R²: {r2:.3f}")
            print(f"    MAE (actual): ₦{mae_actual:,.0f}")
            
        except Exception as e:
            print(f"    ❌ Error training {name}: {e}")
            continue
    
    # Find best model
    if results:
        best_model_name = max(results.keys(), key=lambda x: results[x]['R²'])
        best_model = trained_models[best_model_name]
        print(f"\n🏆 Best Model: {best_model_name} (R²: {results[best_model_name]['R²']:.3f})")
        
        # Save the models
        print("\n💾 Saving models...")
        
        joblib.dump(best_model, 'best_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(label_encoders, 'label_encoders.pkl')
        joblib.dump(list(X.columns), 'feature_names.pkl')
        
        print("✅ Models saved successfully!")
        
        # Test the saved model
        print("\n🧪 Testing saved model...")
        test_model = joblib.load('best_model.pkl')
        test_scaler = joblib.load('scaler.pkl')
        test_encoders = joblib.load('label_encoders.pkl')
        
        # Create a test prediction
        test_input = pd.DataFrame({
            'Age': [25],
            'Gender': [test_encoders['Gender'].transform(['Male'])[0]],
            'State_of_Origin': [test_encoders['State_of_Origin'].transform(['Lagos'])[0]],
            'Region': [test_encoders['Region'].transform(['South'])[0]],
            'Urban_or_Rural': [test_encoders['Urban_or_Rural'].transform(['Urban'])[0]],
            'Household_Income_Bracket': [test_encoders['Household_Income_Bracket'].transform(['Middle'])[0]],
            'Field_of_Study': [test_encoders['Field_of_Study'].transform(['Engineering'])[0]],
            'University_Type': [test_encoders['University_Type'].transform(['Federal'])[0]],
            'GPA_or_Class_of_Degree': [test_encoders['GPA_or_Class_of_Degree'].transform(['Second Class Upper'])[0]],
            'Has_Postgrad_Degree': [test_encoders['Has_Postgrad_Degree'].transform(['No'])[0]],
            'Years_Since_Graduation': [2],
            'Employment_Status': [test_encoders['Employment_Status'].transform(['Employed'])[0]],
            'Salary_Level': [test_encoders['Salary_Level'].transform(['Medium'])[0]]
        })
        
        if best_model_name == 'Linear Regression':
            test_input_scaled = test_scaler.transform(test_input)
            log_pred = test_model.predict(test_input_scaled)[0]
        else:
            log_pred = test_model.predict(test_input)[0]
        
        actual_pred = np.exp(log_pred)
        print(f"Test prediction: ₦{actual_pred:,.2f}")
        print("✅ Model test successful!")
        
        return True
    else:
        print("❌ No models trained successfully!")
        return False

if __name__ == "__main__":
    success = create_working_models()
    if success:
        print("\n🎉 All models created successfully!")
        print("🚀 Ready for deployment!")
    else:
        print("\n❌ Failed to create models!")
