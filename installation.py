import os
import subprocess

os.system("pip install -r requirements.txt")

subprocess.run(["pip", "install", "git+https://github.com/calumholker/pyzx.git"], check=True)

script_dir = os.path.dirname(os.path.abspath(__file__))

target_dir = os.path.join(script_dir, "rl-zx", "gym-zx")
os.chdir(target_dir)
subprocess.run(["pip", "install", "-e", "."], check=True)