"""External-generalization PDE providers."""

from .allen_cahn import AllenCahn1D
from .darcy import DarcyFlow2D
from .kovasznay import KovasznayFlow2D
from .shallow_water import ShallowWater2D

__all__ = [
    "AllenCahn1D",
    "DarcyFlow2D",
    "KovasznayFlow2D",
    "ShallowWater2D",
]
