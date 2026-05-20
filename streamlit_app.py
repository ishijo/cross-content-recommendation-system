"""
Streamlit Cloud entry point.

Streamlit Cloud requires the entry point at the repo root.
This file adds src/ to the Python path and runs app/main.py.

Deploy settings:
  Main file: streamlit_app.py
  Python: 3.11
"""
import sys
import os
from pathlib import Path

_root = Path(__file__).parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))
os.chdir(_root)

exec(open(_root / "app" / "main.py").read())
