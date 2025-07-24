import os
import subprocess
import webbrowser
import time
import fnmatch
import glob

# find nbs
notebooks = [f for f in os.listdir(".")
             if fnmatch.fnmatch(f, "testbench_*.ipynb")]

if not notebooks:
    print("No matching notebooks found.")
    exit(1)

# run nbs
for nb in notebooks:
    print(f"Executing notebook: {nb}")
    subprocess.run([
        "jupyter", "nbconvert", "--to", "notebook", "--execute",
        "--inplace", os.path.join(".", nb)
    ], check=True)

# find profs and open
prof_files = glob.glob(os.path.join("profiling", "**", "*.prof"), recursive=True)
for i, prof in enumerate(prof_files):
    port = 8000 + i
    print(f"Launching {prof} on http://localhost:{port}/snakeviz/")
    
    subprocess.Popen(["snakeviz", "-p", str(port), prof])
    time.sleep(1)
    webbrowser.open(f"http://localhost:{port}/snakeviz/")