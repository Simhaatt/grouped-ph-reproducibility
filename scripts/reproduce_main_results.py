"""Minimal offline reproduction: aggregate tables, figures and verification."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
for script in ['reproduce_tables.py','reproduce_figures.py','verify_repository.py']:
    subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,check=True)
