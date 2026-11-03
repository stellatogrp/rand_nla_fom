"""Finite, shared stopping budgets for every optimization task."""

import math


def budgets(config):
    values = dict(
        time_limit=config.get("time_limit", 1800.0),
        max_inner=config.get("max_inner", 1_000_000),
        max_outer=config.get("max_outer", 50_000),
        inner_time_limit=config.get("inner_time_limit", 300.0),
        max_inner_per_outer=config.get("max_inner_per_outer", 100_000),
    )
    for name, value in values.items():
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
        if name.startswith("max_") and int(value) != value:
            raise ValueError(f"{name} must be an integer")
        if name.startswith("max_"):
            values[name] = int(value)
    return values
