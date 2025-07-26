import os
import subprocess
import webbrowser
import time
import glob

# find profs and open
prof_files = glob.glob(os.path.join("profiling", "**", "*.prof"), recursive=True)
for i, prof in enumerate(prof_files):
    port = 8000 + i
    print(f"Launching {prof} on http://localhost:{port}/snakeviz/")
    
    subprocess.Popen(["snakeviz", "-p", str(port), prof])
    time.sleep(1)
    webbrowser.open(f"http://localhost:{port}/snakeviz/")