"""Register AMPCliff_FLaG_release as the AMPCliff package for tests."""
import sys
import types
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT.parent))

ampcliff_pkg = types.ModuleType("AMPCliff")
ampcliff_pkg.__path__ = [str(RELEASE_ROOT)]
sys.modules["AMPCliff"] = ampcliff_pkg
