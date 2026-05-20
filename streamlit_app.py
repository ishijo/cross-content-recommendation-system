"""
Streamlit Cloud entry point.

Streamlit Cloud requires the entry point at the repo root.
This file adds src/ to the Python path and runs app/main.py.

Deploy settings:
  Main file: streamlit_app.py
  Python: 3.11
"""
import os
import sys
from pathlib import Path

# Ensure src/ is on the path so all internal imports resolve
_root = Path(__file__).parent
sys.path.insert(0, str(_root / "src"))

# Also expose the project root so relative paths in main.py work
os.chdir(str(_root))

# Run the app
exec(open(_root / "app" / "main.py").read())
