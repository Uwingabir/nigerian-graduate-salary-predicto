#!/bin/bash

# Simple deployment script for Railway/Render
# This script prepares the API for deployment

echo "🚀 Preparing Nigerian Graduate Salary Prediction API for deployment..."

# Check if all required files exist
required_files=("main.py" "prediction.py" "requirements.txt" "Dockerfile")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: Required file $file not found!"
        exit 1
    fi
done

# Check if model files exist
model_files=("best_model.pkl" "scaler.pkl" "label_encoders.pkl" "feature_names.pkl")
for file in "${model_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: Model file $file not found!"
        echo "Please run the training script first to generate model files."
        exit 1
    fi
done

echo "✅ All required files found!"
echo "📦 API is ready for deployment!"
echo ""
echo "Deployment options:"
echo "1. Railway: Push to GitHub and connect to Railway"
echo "2. Render: Push to GitHub and connect to Render"
echo "3. Heroku: Use heroku CLI to deploy"
echo "4. Docker: docker build -t salary-api . && docker run -p 8000:8000 salary-api"
echo ""
echo "🌟 Don't forget to update the Flutter app with the deployed API URL!"
