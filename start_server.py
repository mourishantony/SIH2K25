"""
Patient Contact Tracing System - Startup Script
Run this to start both the FastAPI backend server.

For frontend, navigate to /frontend and run: npm run dev
"""
import subprocess
import sys
import os

def main():
    # Change to project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    # Determine the Python executable (prefer venv if exists)
    venv_python = os.path.join(project_root, "venv", "Scripts", "python.exe")
    if os.path.exists(venv_python):
        python_exe = venv_python
    else:
        python_exe = sys.executable
    
    print("=" * 60)
    print("Patient Contact Tracing System")
    print("=" * 60)
    print()
    print(f"Using Python: {python_exe}")
    print()
    print("Starting FastAPI Backend Server...")
    print("API will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    print()
    print("For frontend, open another terminal and run:")
    print("  cd frontend && npm install && npm run dev")
    print()
    print("-" * 60)
    
    # Start uvicorn
    subprocess.run([
        python_exe, "-m", "uvicorn",
        "backend.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])

if __name__ == "__main__":
    main()
