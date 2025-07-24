#!/bin/bash

# Build Flutter APK for mobile demo
echo "Building Flutter APK for Nigerian Graduate Salary Predictor..."

cd "/home/caline/Desktop/ML nigeria/linear_regression_model/summative/FlutterApp/graduate_salary_predictor"

# Clean previous builds
flutter clean

# Get dependencies
flutter pub get

# Build APK
flutter build apk --release

echo "APK built successfully!"
echo "Location: build/app/outputs/flutter-apk/app-release.apk"
echo "You can install this APK on Android devices for the video demo"
