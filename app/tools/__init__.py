from . import device_tools, memory_tools, science, web  # noqa: F401 — l'import enregistre les outils
from .device_tools import DEVICE_TOOL_NAMES  # noqa: F401
from .registry import REGISTRY, Tool, ToolContext, label_for, run, specs  # noqa: F401
