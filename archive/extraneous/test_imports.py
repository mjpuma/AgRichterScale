import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from agririchter.core.config import Config
    from agririchter.visualization.agririchter_scale import AgriRichterScaleVisualizer
    from agririchter.visualization.hp_envelope import HPEnvelopeVisualizer
    print("Imports successful!")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

