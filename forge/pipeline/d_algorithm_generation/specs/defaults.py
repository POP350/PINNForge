"""Necessary execution defaults for the open AlgorithmSpec contract."""

from __future__ import annotations

from .algorithm_spec import algorithm_spec_template


DEFAULT_ALGORITHM_SPEC = algorithm_spec_template()

COMPONENT_DEFAULTS = {
    "activation": {"name": "tanh", "parameters": {}},
    "output_activation": {"name": "identity", "parameters": {}},
    "initialization": {"name": "xavier_normal", "parameters": {}},
    "input_transform": {"name": "none", "parameters": {}},
    "residual_connections": {"enabled": False, "parameters": {}},
    "scheduler": {"name": "none", "parameters": {}},
    "gradient_clipping": {"enabled": False, "parameters": {}},
    "early_stopping": {"enabled": False, "parameters": {}},
}
