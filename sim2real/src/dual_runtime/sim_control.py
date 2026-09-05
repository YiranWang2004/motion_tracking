"""Compatibility imports; simulation and hardware use the same control workflow."""
from .interactive_control import InteractiveDualCoordinator, load_default_command

__all__ = ["InteractiveDualCoordinator", "load_default_command"]
