import requests
import json

# Test if backend server is running
try:
    response = requests.get('http://127.0.0.1:8000/')
    print("Root endpoint response:", response.json())
    
    response = requests.get('http://127.0.0.1:8000/api/algorithms')
    print("Algorithms endpoint response:", json.dumps(response.json(), indent=2))
    print("\n✅ Backend is running correctly!")
except Exception as e:
    print(f"❌ Error connecting to backend: {e}")
    print("\nPlease make sure the backend server is running with:")
    print("  python main.py")
    print("  or")
    print("  python -m uvicorn main:app --reload")
