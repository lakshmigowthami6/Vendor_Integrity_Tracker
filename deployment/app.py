import os
import sys

# Get the path to the root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add root directory to sys.path so imports work correctly
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Path to the root app.py
root_app_path = os.path.join(root_dir, "app.py")

# Execute the root app.py, setting __file__ to the root app path
globals_dict = globals().copy()
globals_dict["__file__"] = root_app_path

with open(root_app_path, "r", encoding="utf-8") as f:
    exec(f.read(), globals_dict)
