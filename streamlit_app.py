"""
Streamlit Cloud entry point.

Streamlit Cloud requires the entry point at the repo root.
This file sets up sys.path and runs app/main.py.

Deploy settings:
  Main file: streamlit_app.py
  Python: 3.11
"""
from pathlib import Path
import sys
import os

_root = Path(__file__).parent

# Add all necessary paths so every import in app/main.py resolves
sys.path.insert(0, str(_root / "src"))   # from models.x import ...  /  from utils.x import ...
sys.path.insert(0, str(_root))           # import project_config  (lives at src/, but also needed at root)
sys.path.insert(0, str(_root / "app"))   # any app-relative imports

os.chdir(str(_root))

# Compile with the correct filename so that __file__ inside app/main.py
# resolves to <repo_root>/app/main.py — not streamlit_app.py.
# This makes Path(__file__).parent.parent == PROJECT_ROOT work correctly.
_app_file = str(_root / "app" / "main.py")
_code = compile(open(_app_file).read(), _app_file, "exec")
exec(_code)
