@echo off
echo Starting Backend Server...
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
