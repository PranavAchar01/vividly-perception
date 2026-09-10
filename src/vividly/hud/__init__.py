"""Compositor-side drawing for the analysis layer.

Analyzers decide; this package only draws what they decided. Nothing in here reads a
frame, keeps state, or imports a script. The compositor owns those.
"""

from vividly.hud.person_red import draw_person_red, person_red_banner
from vividly.hud.substance import draw_substance, substance_banner

__all__ = ["draw_person_red", "draw_substance", "person_red_banner", "substance_banner"]
