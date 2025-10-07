#!/usr/bin/env python3
"""
Setup verification script for Steganography System
"""
import sys
import subprocess

def check_module(module_name, import_name=None):
    """Check if a Python module is installed"""
    if import_name is None:
        import_name = module_name
    try:
        __import__(import_name)
        print(f"✅ {module_name} is installed")
        return True
    except ImportError:
        print(f"❌ {module_name} is NOT installed")
        return False

def main():
    print("=" * 60)
    print("Steganography System - Setup Verification")
    print("=" * 60)
    print()
    
    print("Checking Python modules...")
    print("-" * 60)
    
    modules = [
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
        ('pycryptodome', 'Crypto'),
        ('pillow', 'PIL'),
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'),
        ('PyWavelets', 'pywt'),
    ]
    
    all_installed = True
    for module_name, import_name in modules:
        if not check_module(module_name, import_name):
            all_installed = False
    
    print()
    print("=" * 60)
    
    if all_installed:
        print("✅ All required modules are installed!")
        print()
        print("Next steps:")
        print("1. Open TWO terminal windows")
        print("2. In Terminal 1, run: start_backend.bat")
        print("3. In Terminal 2, run: start_frontend.bat")
        print("4. Open browser to: http://localhost:5173")
    else:
        print("❌ Some modules are missing!")
        print()
        print("To install missing modules, run:")
        print("  cd backend")
        print("  pip install -r ../requirements.txt")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
