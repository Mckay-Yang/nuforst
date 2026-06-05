import sys
from scripts.full_scene_reconstruction import pipeline as _module

sys.modules[__name__] = _module
