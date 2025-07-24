#!/usr/bin/env python3
"""
Nigerian Graduate Salary Prediction Model Training Script
This script implements the complete machine learning pipeline for predicting graduate salaries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 60)
    print("NIGERIAN GRADUATE EMPLOYMENT SALARY PREDICTION")
    print("=" * 60)
    
    # 1. Load and explore data
    print("\n1. Loading and exploring data...")
    df = pd.read_csv('Nigerian_Graduate_Survey_with_Salary.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # 2. Data preprocessing
    print("\n2. Preprocessing data...")
    df_processed = df.copy()
    
    # Remove unemployed graduates as they have 0 salary
    df_processed = df_processed[df_processed['Employment_Status'] == 'Employed'].copy()
    print(f"Dataset after removing unemployed: {df_processed.shape}")
    
    # Drop columns that won't be useful for prediction
    columns_to_drop = ['Graduate_ID', 'Employment_Status', 'Salary_Level', 'State_of_Origin']
    df_processed = df_processed.drop(columns=columns_to_drop)
    
    # Convert categorical variables to numerical
    label_encoders = {}
    categorical_columns = ['Gender', 'Region', 'Urban_or_Rural', 'Household_Income_Bracket', 
                          'Field_of_Study', 'University_Type', 'GPA_or_Class_of_Degree', 'Has_Postgrad_Degree']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
        print(f"{col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # 3. Prepare features and target
    print("\n3. Preparing features and target...")
    X = df_processed.drop('Net_Salary', axis=1)
    y = df_processed['Net_Salary']
    
    print(f"Features: {X.columns.tolist()}")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # 4. Split the data
    print("\n4. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # 5. Standardize features
    print("\n5. Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Train and evaluate models
    print("\n6. Training and evaluating models...")
    models = {
        'Linear Regression': LinearRegression(),
        'SGD Regressor (Gradient Descent)': SGDRegressor(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100)
    }
    
    model_results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for linear models, original for tree-based models
        if 'Linear' in name or 'SGD' in name:
            model.fit(X_train_scaled, y_train)
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        model_results[name] = {
            'model': model,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
        
        print(f"Train MSE: {train_mse:.2f}")
        print(f"Test MSE: {test_mse:.2f}")
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
    
    # 7. Model comparison
    print("\n7. Model comparison:")
    comparison_df = pd.DataFrame({
        'Model': list(model_results.keys()),
        'Train MSE': [results['train_mse'] for results in model_results.values()],
        'Test MSE': [results['test_mse'] for results in model_results.values()],
        'Train R²': [results['train_r2'] for results in model_results.values()],
        'Test R²': [results['test_r2'] for results in model_results.values()]
    })
    
    print(comparison_df)
    
    # Find best model based on test R²
    best_model_idx = comparison_df['Test R²'].idxmax()
    best_model_name = comparison_df.loc[best_model_idx, 'Model']
    best_model = model_results[best_model_name]['model']
    
    print(f"\nBest performing model: {best_model_name}")
    print(f"Best model Test R²: {model_results[best_model_name]['test_r2']:.4f}")
    
    # 8. Generate visualizations
    print("\n8. Generating visualizations...")
    
    # Create plots directory if it doesn't exist
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Set style
    plt.style.use('ggplot')
    sns.set_palette("husl")
    
    # Plot 1: Model comparison
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    x_pos = np.arange(len(comparison_df))
    plt.bar(x_pos - 0.2, comparison_df['Train MSE'], 0.4, label='Train MSE', alpha=0.7)
    plt.bar(x_pos + 0.2, comparison_df['Test MSE'], 0.4, label='Test MSE', alpha=0.7)
    plt.xlabel('Models')
    plt.ylabel('MSE')
    plt.title('Mean Squared Error Comparison')
    plt.xticks(x_pos, [name.split(' ')[0] for name in comparison_df['Model']], rotation=45)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.bar(x_pos - 0.2, comparison_df['Train R²'], 0.4, label='Train R²', alpha=0.7)
    plt.bar(x_pos + 0.2, comparison_df['Test R²'], 0.4, label='Test R²', alpha=0.7)
    plt.xlabel('Models')
    plt.ylabel('R² Score')
    plt.title('R² Score Comparison')
    plt.xticks(x_pos, [name.split(' ')[0] for name in comparison_df['Model']], rotation=45)
    plt.legend()
    
    # Loss curves for SGD
    plt.subplot(2, 2, 3)
    train_losses = []
    test_losses = []
    
    for i in range(1, 101, 10):
        sgd_temp = SGDRegressor(random_state=42, max_iter=i)
        sgd_temp.fit(X_train_scaled, y_train)
        train_pred = sgd_temp.predict(X_train_scaled)
        test_pred = sgd_temp.predict(X_test_scaled)
        train_losses.append(mean_squared_error(y_train, train_pred))
        test_losses.append(mean_squared_error(y_test, test_pred))
    
    iterations = list(range(1, 101, 10))
    plt.plot(iterations, train_losses, label='Training Loss', marker='o')
    plt.plot(iterations, test_losses, label='Test Loss', marker='s')
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.title('SGD Training vs Test Loss Curve')
    plt.legend()
    
    # Prediction scatter plot for best model
    plt.subplot(2, 2, 4)
    best_results = model_results[best_model_name]
    plt.scatter(y_test, best_results['y_test_pred'], alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Salary')
    plt.ylabel('Predicted Salary')
    plt.title(f'{best_model_name.split(" ")[0]} - Actual vs Predicted')
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Data distribution analysis
    plt.figure(figsize=(15, 10))
    
    # Distribution of target variable
    plt.subplot(2, 3, 1)
    plt.hist(df['Net_Salary'], bins=50, alpha=0.7, color='skyblue')
    plt.title('Distribution of Net Salary')
    plt.xlabel('Net Salary (₦)')
    plt.ylabel('Frequency')
    
    # Employment status
    plt.subplot(2, 3, 2)
    df['Employment_Status'].value_counts().plot(kind='bar', color='lightcoral')
    plt.title('Employment Status Distribution')
    plt.xticks(rotation=45)
    
    # Salary by gender
    plt.subplot(2, 3, 3)
    sns.boxplot(data=df[df['Employment_Status'] == 'Employed'], x='Gender', y='Net_Salary')
    plt.title('Salary by Gender')
    plt.xticks(rotation=45)
    
    # Salary by field of study
    plt.subplot(2, 3, 4)
    sns.boxplot(data=df[df['Employment_Status'] == 'Employed'], x='Field_of_Study', y='Net_Salary')
    plt.title('Salary by Field of Study')
    plt.xticks(rotation=45)
    
    # Salary by university type
    plt.subplot(2, 3, 5)
    sns.boxplot(data=df[df['Employment_Status'] == 'Employed'], x='University_Type', y='Net_Salary')
    plt.title('Salary by University Type')
    plt.xticks(rotation=45)
    
    # Age vs Salary
    plt.subplot(2, 3, 6)
    employed_df = df[df['Employment_Status'] == 'Employed']
    plt.scatter(employed_df['Age'], employed_df['Net_Salary'], alpha=0.6)
    plt.xlabel('Age')
    plt.ylabel('Net Salary (₦)')
    plt.title('Age vs Salary')
    
    plt.tight_layout()
    plt.savefig('plots/data_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 9. Save models and preprocessing objects
    print("\n9. Saving models and preprocessing objects...")
    
    # Save models
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(model_results['Linear Regression']['model'], 'linear_regression_model.pkl')
    joblib.dump(model_results['Decision Tree']['model'], 'decision_tree_model.pkl')
    joblib.dump(model_results['Random Forest']['model'], 'random_forest_model.pkl')
    
    # Save preprocessing objects
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(X.columns.tolist(), 'feature_names.pkl')
    
    # 10. Test prediction function
    print("\n10. Testing prediction function...")
    
    def predict_salary(age, gender, region, urban_or_rural, household_income_bracket, 
                      field_of_study, university_type, gpa_or_class_of_degree, 
                      has_postgrad_degree, years_since_graduation):
        """Test prediction function"""
        
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
        for col in categorical_columns:
            input_data[col] = label_encoders[col].transform(input_data[col])
        
        # Scale features if the best model requires it
        if best_model_name in ['Linear Regression', 'SGD Regressor (Gradient Descent)']:
            input_scaled = scaler.transform(input_data)
            prediction = best_model.predict(input_scaled)[0]
        else:
            prediction = best_model.predict(input_data)[0]
        
        return max(0, prediction)  # Ensure non-negative salary
    
    # Test prediction
    test_prediction = predict_salary(
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
    
    # 11. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Dataset: {df.shape[0]} graduates surveyed")
    print(f"Employed graduates used for modeling: {df_processed.shape[0]}")
    print(f"Features used: {len(X.columns)}")
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Model Test R²: {model_results[best_model_name]['test_r2']:.4f}")
    print(f"Best Model Test MSE: {model_results[best_model_name]['test_mse']:.2f}")
    
    print("\nFiles saved:")
    print("- best_model.pkl")
    print("- linear_regression_model.pkl") 
    print("- decision_tree_model.pkl")
    print("- random_forest_model.pkl")
    print("- scaler.pkl")
    print("- label_encoders.pkl")
    print("- feature_names.pkl")
    print("- plots/model_comparison.png")
    print("- plots/data_analysis.png")
    
    print("\nReady for API deployment!")
    return best_model_name, model_results[best_model_name]['test_r2']

if __name__ == "__main__":
    best_model, best_score = main()
