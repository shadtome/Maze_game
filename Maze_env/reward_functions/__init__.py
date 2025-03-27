import os
import importlib

# Get all Python files in the current folder (excluding __init__.py)
reward_fun_files = [
    f[:-3] for f in os.listdir(os.path.dirname(__file__)) 
    if f.endswith(".py") and f != "__init__.py"
]

# Import each game file dynamically
for reward_fun in reward_fun_files:
    module = importlib.import_module(f".{reward_fun}", package=__name__)
    globals()[reward_fun] = module  # Add to global namespace