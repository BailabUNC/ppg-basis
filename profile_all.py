import os
import subprocess
import webbrowser
import time
import fnmatch
import glob

# === Config ===
NOTEBOOK_DIR = "."
PROFILE_DIR = "profile_output"
NOTEBOOK_PATTERN = "testbench_*.ipynb"
START_PORT = 8000
DELAY_BETWEEN = 1  # seconds between launching snakeviz tabs

# === Step 1: Find notebooks matching pattern ===
notebooks = [f for f in os.listdir(NOTEBOOK_DIR)
             if fnmatch.fnmatch(f, NOTEBOOK_PATTERN)]
notebooks.sort()

if not notebooks:
    print("No matching notebooks found.")
    exit(1)

# === Step 2: Execute notebooks ===
for nb in notebooks:
    print(f"Executing notebook: {nb}")
    subprocess.run([
        "jupyter", "nbconvert", "--to", "notebook", "--execute",
        "--inplace", os.path.join(NOTEBOOK_DIR, nb)
    ], check=True)

# === Step 3: Find .prof files ===
profile_dir = "profile_output"
prof_files = glob.glob(os.path.join(profile_dir, "**", "*.prof"), recursive=True)

if not prof_files:
    print("No .prof files found in profile_output/")
    exit(1)

# === Step 4: Launch snakeviz in browser tabs ===
for i, prof in enumerate(prof_files):
    port = START_PORT + i
    filepath = os.path.join(PROFILE_DIR, prof)
    print(f"Launching {filepath} on http://localhost:{port}")
    
    subprocess.Popen(["snakeviz", "-p", str(port), filepath])
    time.sleep(DELAY_BETWEEN)
    webbrowser.open(f"http://localhost:{port}")