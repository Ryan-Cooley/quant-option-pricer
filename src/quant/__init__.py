"""
Quantitative Finance Package

Core pricing and risk analysis functions for options and derivatives.
"""

from .pricing import (
    d1_d2,
    bs_price,
    bs_delta,
    bs_vega,
    mc_euro_price,
)

from .hedging import (
    simulate_gbm_paths,
    simulate_delta_hedge,
    HedgeResult,
)

from .iv import (
    implied_vol_bs,
    _bounds_call_put,
)

__all__ = [
    "d1_d2",
    "bs_price",
    "bs_delta",
    "bs_vega",
    "mc_euro_price",
    "simulate_gbm_paths",
    "simulate_delta_hedge",
    "HedgeResult",
    "implied_vol_bs",
    "_bounds_call_put",
]
