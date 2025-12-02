import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from agririchter.core.config import Config
    print("Config import successful!")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

