"""Inner starting points, independent of matrix/sketch/block-sampling RNGs."""

import numpy as np

MODES = ("warm", "fixed", "gaussian")
BASELINES = ("pdhg", "osqp", "scs")


def settings(config):
    mode = config.get("inner_initialization", "warm")
    if mode not in MODES:
        raise ValueError(f"unknown inner initialization: {mode}")
    seed = config.get("initialization_seed", config["seed"] + 200003)
    scale = config.get("gaussian_scale", 1.0)
    fixed = config.get("fixed_initial_value", 0.0)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("initialization_seed must be a nonnegative integer")
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(fixed):
        raise ValueError(
            "initialization requires finite fixed value and positive scale"
        )
    return dict(
        inner_initialization=mode,
        initialization_seed=seed,
        gaussian_scale=scale,
        fixed_initial_value=fixed,
    )


def starting_point(previous, outer, config):
    opts = settings(config)
    mode = opts["inner_initialization"]
    if mode == "warm":
        return previous
    if mode == "fixed":
        return np.full_like(previous, opts["fixed_initial_value"])
    # Index by outer iteration so resuming never changes subsequent draws.
    # This stream cannot consume random numbers used for solver setup or steps.
    rng = np.random.default_rng(
        np.random.SeedSequence([opts["initialization_seed"], int(outer)])
    )
    return rng.normal(size=previous.size) * (
        opts["gaussian_scale"] / np.sqrt(previous.size)
    )


def variants(config, modes=MODES, gaussian_scale=1.0, fixed_initial_value=0.0):
    if config["inner"] in BASELINES:
        yield config  # no exposed inner starting point; one baseline per instance
        return
    for mode in modes:
        yield dict(
            config,
            inner_initialization=mode,
            initialization_seed=config["seed"] + 200003,
            gaussian_scale=gaussian_scale,
            fixed_initial_value=fixed_initial_value,
        )
