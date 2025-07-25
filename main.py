"""
Deployment wrapper for Render.com
This file ensures Render can find and run the FastAPI application
"""
import sys
import os

# Add the API directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'summative', 'API'))

# Import the FastAPI app from the API directory
from summative.API.main import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
