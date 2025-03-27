import os
import importlib

# Get all Python files in the current folder (excluding __init__.py)
game_files = [
    f[:-3] for f in os.listdir(os.path.dirname(__file__)) 
    if f.endswith(".py") and f != "__init__.py"
]

# Import each game file dynamically
for game in game_files:
    module = importlib.import_module(f".{game}", package=__name__)
    globals()[game] = module  # Add to global namespace