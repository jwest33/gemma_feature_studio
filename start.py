#!/usr/bin/env python3
"""
Gemma Feature Studio - Development Server Launcher

Starts both the backend (FastAPI) and frontend (Next.js) services.
Backend starts first and waits for health check before starting frontend.
"""

import subprocess
import sys
import time
import threading
import urllib.request
import urllib.error
import platform
from pathlib import Path


def get_npm_command() -> list[str]:
    """Get the npm command for the current platform.

    On Windows, npm is a batch file (npm.cmd) that requires shell=True
    or explicit .cmd extension. We use shell=True on Windows for PATH resolution.
    """
    return ["npm"]


def get_shell_flag() -> bool:
    """Returns True on Windows (needed for PATH resolution of npm.cmd), False otherwise."""
    return platform.system() == "Windows"

# Configuration
BACKEND_PORT = 8000
FRONTEND_PORT = 3000
BACKEND_HEALTH_URL = f"http://localhost:{BACKEND_PORT}/api/health"
BACKEND_TIMEOUT = 60  # seconds to wait for backend
ROOT_DIR = Path(__file__).parent.resolve()
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"


def check_requirements():
    """Check that required directories and files exist."""
    errors = []

    if not BACKEND_DIR.exists():
        errors.append(f"Backend directory not found: {BACKEND_DIR}")
    if not FRONTEND_DIR.exists():
        errors.append(f"Frontend directory not found: {FRONTEND_DIR}")
    if not (BACKEND_DIR / "main.py").exists():
        errors.append("Backend main.py not found")
    if not (FRONTEND_DIR / "package.json").exists():
        errors.append("Frontend package.json not found")

    if errors:
        print("Error: Missing required files/directories:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)


def check_node_modules():
    """Check if node_modules exists, prompt to install if not."""
    node_modules = FRONTEND_DIR / "node_modules"
    if not node_modules.exists():
        print("Frontend dependencies not installed.")
        print(f"Run: cd {FRONTEND_DIR} && npm install")
        response = input("Install now? [Y/n] ").strip().lower()
        if response in ("", "y", "yes"):
            print("Installing frontend dependencies...")
            subprocess.run(
                get_npm_command() + ["install"],
                cwd=FRONTEND_DIR,
                shell=get_shell_flag(),
                check=True,
            )
        else:
            print("Please install dependencies and try again.")
            sys.exit(1)


def start_backend():
    """Start the FastAPI backend server."""
    print(f"Starting backend server on http://localhost:{BACKEND_PORT}...")

    python_exe = sys.executable

    return subprocess.Popen(
        [python_exe, "-m", "uvicorn", "main:app",
         "--host", "0.0.0.0",
         "--port", str(BACKEND_PORT),
         "--reload"],
        cwd=BACKEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def wait_for_backend(process, timeout=BACKEND_TIMEOUT):
    """Wait for backend to be healthy by polling the health endpoint."""
    print(f"Waiting for backend to be ready (timeout: {timeout}s)...")

    start_time = time.time()
    check_interval = 0.5

    while time.time() - start_time < timeout:
        # Check if process has crashed
        if process.poll() is not None:
            print("Error: Backend process exited unexpectedly")
            return False

        try:
            req = urllib.request.Request(BACKEND_HEALTH_URL, method='GET')
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    elapsed = time.time() - start_time
                    print(f"Backend ready! (took {elapsed:.1f}s)")
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            pass

        # Print progress dot every 2 seconds
        elapsed = time.time() - start_time
        if int(elapsed) % 2 == 0 and int(elapsed) > 0:
            print(".", end="", flush=True)

        time.sleep(check_interval)

    print(f"\nError: Backend did not become ready within {timeout} seconds")
    return False


def start_frontend():
    """Start the Next.js frontend development server."""
    print(f"Starting frontend server on http://localhost:{FRONTEND_PORT}...")

    return subprocess.Popen(
        get_npm_command() + ["run", "dev"],
        cwd=FRONTEND_DIR,
        shell=get_shell_flag(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def stream_output(process, prefix):
    """Stream output from a process with a prefix."""
    try:
        for line in iter(process.stdout.readline, ""):
            if line:
                print(f"[{prefix}] {line.rstrip()}")
    except Exception:
        pass


def main():
    print("=" * 60)
    print("  Gemma Feature Studio - Development Server")
    print("=" * 60)
    print()

    check_requirements()
    check_node_modules()

    processes = []

    try:
        # Start backend first
        backend_proc = start_backend()
        processes.append(("backend", backend_proc))

        # Start streaming backend output in background
        backend_thread = threading.Thread(
            target=stream_output,
            args=(backend_proc, "backend"),
            daemon=True
        )
        backend_thread.start()

        # Wait for backend to be healthy
        if not wait_for_backend(backend_proc):
            print("Failed to start backend. Check the logs above for errors.")
            raise KeyboardInterrupt

        print()

        # Now start frontend
        frontend_proc = start_frontend()
        processes.append(("frontend", frontend_proc))

        # Start streaming frontend output
        frontend_thread = threading.Thread(
            target=stream_output,
            args=(frontend_proc, "frontend"),
            daemon=True
        )
        frontend_thread.start()

        print()
        print("=" * 60)
        print(f"  Backend API:  http://localhost:{BACKEND_PORT}")
        print(f"  Frontend UI:  http://localhost:{FRONTEND_PORT}")
        print(f"  API Docs:     http://localhost:{BACKEND_PORT}/docs")
        print("=" * 60)
        print()
        print("Press Ctrl+C to stop all services...")
        print()

        # Wait for processes
        while True:
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"\n{name} process exited with code {proc.returncode}")
                    raise KeyboardInterrupt
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nShutting down services...")

    finally:
        # Terminate all processes
        for name, proc in processes:
            if proc.poll() is None:
                print(f"Stopping {name}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

        print("All services stopped.")


if __name__ == "__main__":
    main()
