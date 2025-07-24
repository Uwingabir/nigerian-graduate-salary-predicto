#!/bin/bash

# Repository Setup Script
# Replace 'nigerian-graduate-salary-predictor' with your chosen name

REPO_NAME="nigerian-graduate-salary-predictor"
GITHUB_USERNAME="your-username"  # Replace with your GitHub username

echo "Setting up repository: $REPO_NAME"

# Initialize git repository
cd "/home/caline/Desktop/ML nigeria/linear_regression_model"
git init

# Add all files
git add .

# Initial commit
git commit -m "Initial commit: Nigerian Graduate Salary Prediction ML Platform

- Complete linear regression analysis with gradient descent
- FastAPI web service with CORS and validation
- Flutter mobile app with prediction interface
- Comprehensive documentation and deployment files"

# Add remote origin (you'll need to create the repo on GitHub first)
echo "Next steps:"
echo "1. Create repository '$REPO_NAME' on GitHub"
echo "2. Run: git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
echo "3. Run: git branch -M main"
echo "4. Run: git push -u origin main"
