# Render Deployment Guide

## Your app is ready for deployment! 🚀

### Files Ready:
- ✅ `render.yaml` - Deployment configuration
- ✅ `summative/API/main.py` - FastAPI application
- ✅ `summative/API/requirements.txt` - Dependencies
- ✅ Rule-based prediction system (no large ML files)

### Quick Deployment Steps:

#### Option 1: Deploy from GitHub (Recommended)
1. Create a new repository on your GitHub account
2. Push this code to your repository:
   ```bash
   git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```
3. Go to [render.com](https://render.com)
4. Click "New +" → "Web Service"
5. Connect your GitHub repository
6. Render will automatically detect the `render.yaml` file

#### Option 2: Manual Configuration
If you can't use GitHub, manually configure on Render:

**Basic Settings:**
- Name: `nigerian-graduate-salary-predictor`
- Environment: `Python`
- Region: Choose closest to your users
- Branch: `main`

**Build & Deploy:**
- Build Command: 
  ```
  cd summative/API && rm -f *.pkl && pip install --no-cache-dir --upgrade pip setuptools wheel && pip install --no-cache-dir -r requirements.txt && python3 -c "import joblib; print('Joblib version:', joblib.__version__)"
  ```
- Start Command: 
  ```
  cd summative/API && uvicorn main:app --host 0.0.0.0 --port $PORT
  ```
- Python Version: `3.9`

**Environment Variables:**
No special environment variables needed for basic deployment.

### After Deployment:

1. Your API will be available at: `https://your-app-name.onrender.com`

2. Test endpoints:
   - `GET /` - Root endpoint
   - `GET /health` - Health check
   - `GET /docs` - API documentation
   - `POST /predict` - Salary prediction

3. Use the provided `test_deployment.py` script to verify everything works

### API Documentation:
Once deployed, visit `https://your-app-name.onrender.com/docs` for interactive API documentation.

### Sample API Request:
```json
POST /predict
{
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
```

### Troubleshooting:
- Check build logs if deployment fails
- Ensure all dependencies are in requirements.txt
- Verify Python version compatibility
- Check that the start command points to the correct directory

Good luck with your deployment! 🎉
