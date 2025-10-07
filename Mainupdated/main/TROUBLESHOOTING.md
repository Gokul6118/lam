# Troubleshooting Guide - Steganography System

## Issue: "Failed to load algorithms" on Embed Page

This issue occurs when the frontend cannot connect to the backend server. Follow these steps to resolve it:

### Step 1: Fix the Crypto Module Issue

The backend requires `pycryptodome` to be properly installed. Run:

```bash
cd backend
pip uninstall crypto -y
pip install --force-reinstall pycryptodome
```

### Step 2: Start the Backend Server

Open a **new terminal** and run:

```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or simply double-click `start_backend.bat`

You should see output like:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Step 3: Verify Backend is Running

Open a browser and go to: `http://127.0.0.1:8000/api/algorithms`

You should see JSON output like:
```json
{
  "encryption": ["AES-256", "RSA-2048"],
  "steganography": ["LSB", "PVD", "DCT", "F5"]
}
```

### Step 4: Start the Frontend Server

Open a **second terminal** and run:

```bash
npm run dev
```

Or simply double-click `start_frontend.bat`

You should see output like:
```
VITE v5.x.x  ready in xxx ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
```

### Step 5: Test the Application

1. Open your browser to `http://localhost:5173`
2. Navigate to the Embed page
3. The algorithm dropdowns should now show the available options instead of "Loading..."

## Common Issues

### Issue: Port 8000 is already in use

**Solution:** Kill the process using port 8000:

```bash
# Find the process
netstat -ano | findstr :8000

# Kill it (replace PID with the actual process ID)
taskkill /PID <PID> /F
```

### Issue: Port 5173 is already in use

**Solution:** Kill the process using port 5173 or change the port in `frontend/vite.config.ts`

### Issue: Module not found errors

**Solution:** Install all dependencies:

```bash
# Backend
cd backend
pip install -r ../requirements.txt

# Frontend
cd ..
npm install
```

### Issue: CORS errors in browser console

**Solution:** The backend is already configured to allow CORS. Make sure:
1. Backend is running on port 8000
2. Frontend is running on port 5173
3. The Vite proxy is correctly configured (already fixed)

## Quick Start Commands

### Terminal 1 (Backend):
```bash
cd d:\newver\Mainupdated\main\backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal 2 (Frontend):
```bash
cd d:\newver\Mainupdated\main
npm run dev
```

## Verification Checklist

- [ ] Backend server is running on http://127.0.0.1:8000
- [ ] `/api/algorithms` endpoint returns JSON with encryption and steganography methods
- [ ] Frontend server is running on http://localhost:5173
- [ ] No CORS errors in browser console
- [ ] Algorithm dropdowns show options instead of "Loading..."
