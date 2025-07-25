# Nigerian Graduate Salary Prediction Project

## Mission and Problem (4 lines)
**Mission:** To analyze the factors affecting graduate employment in Nigeria and predict employment outcomes based on education, demographics, and socioeconomic background, helping young Africans make informed career and education decisions.

**Problem:** Many Nigerian graduates struggle to find well-paying jobs despite their education. Our model helps predict employment outcomes based on educational and demographic factors, enabling better career planning and policy decisions.

**Solution:** An AI-powered prediction system that estimates graduate salary and employment probability, helping young Africans make informed education and career choices while identifying factors that improve employment outcomes.

## 🎯 **Assignment Task 2 - COMPLETED**

### ✅ **FastAPI Requirements Met:**
- **FastAPI Framework:** ✅ Implemented with proper structure
- **Pydantic Data Validation:** ✅ BaseModel with type enforcement and range constraints  
- **Uvicorn Server:** ✅ ASGI server running on Render
- **CORS Middleware:** ✅ Cross-Origin Resource Sharing enabled
- **POST Endpoint:** ✅ `/predict` endpoint with structured data validation
- **requirements.txt:** ✅ All dependencies properly specified
- **Render Deployment:** ✅ Successfully hosted on free tier
- **Swagger UI:** ✅ Interactive API documentation available

### 🌐 **Live Deployment Links:**
- **📖 Swagger UI (Submit this):** https://nigerian-graduate-salary-predicto-3.onrender.com/docs
- **🔗 API Base URL:** https://nigerian-graduate-salary-predicto-3.onrender.com

## Project Structure
```
linear_regression_model/
├── summative/
│   ├── linear_regression/
│   │   ├── multivariate.ipynb          # Main notebook with model analysis
│   │   ├── Nigerian_Graduate_Survey_with_Salary.csv  # Dataset
│   │   └── *.pkl files                 # Saved models and preprocessors
│   ├── API/
│   │   ├── main.py                     # FastAPI application
│   │   ├── prediction.py               # Prediction functions
│   │   └── requirements.txt            # Python dependencies
│   └── FlutterApp/
│       └── graduate_salary_predictor/   # Flutter mobile app
└── README.md
```

## Features Implemented

### 📊 Task 1: Linear Regression Analysis
- ✅ Data visualization and interpretation
- ✅ Feature engineering and preprocessing
- ✅ Data standardization and encoding
- ✅ Linear Regression with Gradient Descent (SGD)
- ✅ Model comparison (Linear Regression, Decision Tree, Random Forest)
- ✅ Loss curves and performance visualization
- ✅ Model persistence and prediction functions

### 🚀 Task 2: FastAPI Web Service
- ✅ FastAPI application with CORS middleware
- ✅ Pydantic models for data validation
- ✅ POST endpoint for predictions with type checking
- ✅ Range constraints for all inputs
- ✅ Comprehensive error handling
- ✅ Swagger UI documentation
- ✅ Ready for Render deployment

### 📱 Task 3: Flutter Mobile App
- ✅ Multi-page application (Splash, About, Prediction)
- ✅ 10 input fields for all prediction variables
- ✅ Form validation and type checking
- ✅ HTTP API integration
- ✅ Professional UI design
- ✅ Error handling and loading states

## How to Run the Mobile App

### Prerequisites
- Flutter SDK (3.8.0 or higher)
- Android Studio / VS Code with Flutter plugins
- Android/iOS device or emulator

### Installation Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Uwingabir/nigerian-graduate-salary-predicto.git
   cd nigerian-graduate-salary-predicto/summative/FlutterApp/graduate_salary_predictor
   ```

2. **Install dependencies:**
   ```bash
   flutter pub get
   ```

3. **Update API URL:**
   - Open `lib/main.dart`
   - Replace `https://your-api-url.com/predict` with your actual deployed API URL

4. **Run the app:**
   ```bash
   flutter run
   ```

### For Android APK:
```bash
flutter build apk --release
```
The APK will be in `build/app/outputs/flutter-apk/app-release.apk`

## Model Performance Summary

| Model | Test R² | Test MSE | Notes |
|-------|---------|----------|-------|
| Linear Regression | 0.XXXX | XXXXX | Baseline model |
| Decision Tree | 0.XXXX | XXXXX | Non-linear patterns |
| Random Forest | 0.XXXX | XXXXX | Best performing |
| SGD Regressor | 0.XXXX | XXXXX | Gradient descent implementation |

*Values will be populated after running the notebook*

## API Usage Example

### Request
```bash
curl -X POST "https://nigerian-graduate-salary-predicto-3.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 25,
       "gender": "Male",
       "region": "South",
       "urban_or_rural": "Urban",
       "household_income_bracket": "Middle",
       "field_of_study": "Engineering",
       "university_type": "Federal",
       "gpa_or_class_of_degree": "Second Class Upper",
       "has_postgrad_degree": "Yes",
       "years_since_graduation": 2
     }'
```

### Response
```json
{
  "predicted_salary": 285000.50,
  "formatted_salary": "₦285,000.50",
  "input_data": {...},
  "model_confidence": "Moderate salary range"
}
```

## Video Demo
🎥 **YouTube Demo Link:** [Nigerian Graduate Salary Prediction Demo](https://youtube.com/watch?v=your-video-id)

*Upload your 5-minute demo video and update this link*

## Technologies Used
- **Machine Learning:** Python, scikit-learn, pandas, numpy, matplotlib, seaborn
- **API:** FastAPI, Pydantic, uvicorn
- **Mobile App:** Flutter, Dart, HTTP package
- **Deployment:** Render (API hosting)
- **Data:** Nigerian Graduate Survey Dataset

## License
This project is for educational purposes as part of a machine learning assignment.

---
*Built with ❤️ for Nigerian graduates*
