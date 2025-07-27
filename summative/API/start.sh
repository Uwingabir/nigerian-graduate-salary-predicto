#!/bin/bash
echo "🧹 Cleaning up old model files..."
rm -f *.pkl
echo "🚀 Starting API server..."
uvicorn main:app --host=0.0.0.0 --port=${PORT:-8000}
