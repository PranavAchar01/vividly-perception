"""Compositor-side drawing for the analysis layer.

Analyzers decide; this package only draws what they decided. Nothing in here reads a
frame, keeps state, or imports a script. The compositor owns those.
"""

from vividly.hud.substance import draw_substance, substance_banner

__all__ = ["draw_substance", "substance_banner"]
