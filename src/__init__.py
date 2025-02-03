import sys
from pathlib import Path
root_dir = Path.cwd().resolve().parent
if root_dir.exists():
    if str(root_dir) not in sys.path: sys.path.append(str(root_dir))
else:
    raise FileNotFoundError('Root directory not found')