from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "sentinel"
if PACKAGE_ROOT.exists():
    sys.path.insert(0, str(PACKAGE_ROOT))
