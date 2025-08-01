#!/usr/bin/env python3
"""
FSPEAK Launcher - Start both FastAPI backend and Gradio frontend

This script launches both the FastAPI backend server and the Gradio frontend
interface in separate processes for easy development and testing.
"""

import os
import sys
import time
import signal
import subprocess
import threading
from pathlib import Path


def load_env_file():
    """Load environment variables from .env file if it exists"""
    env_file = Path(".env")
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("✅ Loaded environment variables from .env")
        except ImportError:
            print("⚠️  python-dotenv not installed, skipping .env file")
    else:
        print("⚠️  .env file not found, using default settings")
        print("   Copy .env.example to .env and configure your API keys")


def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "fastapi",
        "uvicorn", 
        "gradio",
        "pydantic",
        "httpx"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("   Install them with: pip install -r requirements.txt")
        return False
    
    print("✅ All required dependencies are installed")
    return True


def check_ffmpeg():
    """Check if FFmpeg is available"""
    ffmpeg_path = os.getenv("FFMPEG_PATH", "C:\\ffmpeg\\bin\\ffmpeg.exe")
    
    if os.path.exists(ffmpeg_path):
        print(f"✅ FFmpeg found at: {ffmpeg_path}")
        return True
    else:
        print(f"⚠️  FFmpeg not found at: {ffmpeg_path}")
        print("   Update FFMPEG_PATH in .env or install FFmpeg")
        return False


def start_backend():
    """Start the FastAPI backend server"""
    host = os.getenv("FAST_API_HOST", "0.0.0.0")
    port = int(os.getenv("FAST_API_PORT", "8000"))
    
    print(f"🚀 Starting FastAPI backend on {host}:{port}")
    
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "main:app",
        "--host", host,
        "--port", str(port),
        "--reload" if os.getenv("DEV_MODE", "false").lower() == "true" else "--no-reload"
    ]
    
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )


def start_frontend():
    """Start the Gradio frontend"""
    print("🎨 Starting Gradio frontend")
    
    cmd = [sys.executable, "gradio_app.py"]
    
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )


def monitor_process(process, name):
    """Monitor a process and print its output"""
    try:
        for line in process.stdout:
            print(f"[{name}] {line.strip()}")
    except Exception as e:
        print(f"❌ Error monitoring {name}: {e}")


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\n🛑 Shutting down FSPEAK...")
    sys.exit(0)


def main():
    """Main launcher function"""
    print("🎬 FSPEAK - FFmpeg Natural Language Interface")
    print("=" * 50)
    
    # Load environment variables
    load_env_file()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check FFmpeg (warning only)
    check_ffmpeg()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    backend_process = None
    frontend_process = None
    
    try:
        # Start backend
        backend_process = start_backend()
        
        # Wait a moment for backend to start
        print("⏳ Waiting for backend to start...")
        time.sleep(3)
        
        # Check if backend started successfully
        if backend_process.poll() is not None:
            print("❌ Backend failed to start")
            return 1
        
        # Start frontend
        frontend_process = start_frontend()
        
        # Wait a moment for frontend to start
        time.sleep(2)
        
        # Check if frontend started successfully
        if frontend_process.poll() is not None:
            print("❌ Frontend failed to start")
            return 1
        
        print("\n✅ FSPEAK is running!")
        print("📊 Backend API: http://localhost:8000")
        print("🎨 Frontend UI: http://localhost:7860")
        print("📚 API Docs: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop")
        
        # Start monitoring threads
        backend_thread = threading.Thread(
            target=monitor_process, 
            args=(backend_process, "Backend"),
            daemon=True
        )
        frontend_thread = threading.Thread(
            target=monitor_process, 
            args=(frontend_process, "Frontend"),
            daemon=True
        )
        
        backend_thread.start()
        frontend_thread.start()
        
        # Wait for processes
        while True:
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("❌ Backend process stopped unexpectedly")
                break
            
            if frontend_process.poll() is not None:
                print("❌ Frontend process stopped unexpectedly")
                break
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n🛑 Received shutdown signal")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    finally:
        # Clean up processes
        if backend_process and backend_process.poll() is None:
            print("🔄 Stopping backend...")
            backend_process.terminate()
            try:
                backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_process.kill()
        
        if frontend_process and frontend_process.poll() is None:
            print("🔄 Stopping frontend...")
            frontend_process.terminate()
            try:
                frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                frontend_process.kill()
        
        print("✅ FSPEAK stopped successfully")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())