#!/usr/bin/env python3
"""
Script to kill the user's own Python processes that are using GPU memory.
This checks the process owner and only kills processes owned by the current user.
"""

import os
import subprocess
import sys

def get_current_user():
    return os.getlogin()

def get_gpu_python_pids():
    """Get PIDs of Python processes using GPU from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        pids = [line.strip() for line in result.stdout.split('\n') if line.strip()]
        return pids
    except subprocess.CalledProcessError:
        print("Error running nvidia-smi. Make sure it's installed.")
        return []

def get_process_user(pid):
    """Get the user of a process."""
    try:
        result = subprocess.run(
            ["ps", "-p", pid, "-o", "user=", "--no-headers"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def kill_process(pid):
    """Kill a process gracefully, then forcefully if needed."""
    try:
        # Try SIGTERM first
        subprocess.run(["kill", pid], check=True)
        print(f"Sent SIGTERM to process {pid}")
    except subprocess.CalledProcessError:
        try:
            # Try SIGKILL
            subprocess.run(["kill", "-9", pid], check=True)
            print(f"Sent SIGKILL to process {pid}")
        except subprocess.CalledProcessError:
            print(f"Failed to kill process {pid}")

def main():
    current_user = get_current_user()
    print(f"Current user: {current_user}")

    gpu_pids = get_gpu_python_pids()
    if not gpu_pids:
        print("No GPU processes found.")
        return

    print(f"GPU processes found: {gpu_pids}")

    for pid in gpu_pids:
        user = get_process_user(pid)
        if user == current_user:
            print(f"Killing your process {pid} (user: {user})")
            kill_process(pid)
        else:
            print(f"Skipping process {pid} (owned by {user}, not you)")

if __name__ == "__main__":
    main()