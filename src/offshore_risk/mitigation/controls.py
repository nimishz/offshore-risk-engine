"""Risk controls as transformations of the epistemic parameters.

A control never edits the model code. It changes parameter values (with an
uncertain effectiveness drawn from its own random stream) or the structure of
a protection system (e.g. number of firewater pumps), and the whole model is
re-run. Risk reduction is then measured, not assumed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import numpy as np

from ..distributions import make_distribution

OPS = ("reduce", "increase", "close_gap", "set", "multiply")


def apply_effect(x: np.ndarray, op: str, e: np.ndarray) -> np.ndarray:
    if op == "reduce":
        return x * (1.0 - e)
    if op == "increase":
        return x * (1.0 + e)
    if op == "close_gap":
        return x + e * (1.0 - x)
    if op == "set":
        return np.broadcast_to(np.asarray(e, dtype=float), np.shape(x)).copy()
    if op == "multiply":
        return x * e
    raise ValueError(f"unknown effect op {op!r}; expected one of {OPS}")


def apply_mitigations(
    theta: Mapping[str, np.ndarray],
    keys: Iterable[str],
    mitigations_cfg: Mapping[str, dict],
    rng_for: Callable[[str], np.random.Generator],
) -> tuple[dict, dict, dict]:
    """Return (modified theta, architecture changes, sampled effectiveness per control)."""
    out = {k: np.array(v, dtype=float, copy=True) for k, v in theta.items()}
    n = len(next(iter(theta.values())))
    arch: dict = {}
    eff: dict = {}
    for key in keys:
        m = mitigations_cfg[key]
        e = make_distribution(m.get("effectiveness", {"type": "fixed", "value": 1.0})).ppf(rng_for(f"mitigation:{key}").random(n))
        eff[key] = e
        for fx in m.get("effects", []):
            w = float(fx.get("weight", 1.0))
            out[fx["param"]] = apply_effect(out[fx["param"]], fx["op"], w * e)
        for a, v in (m.get("architecture") or {}).items():
            arch[a] = v
    return out, arch, eff


def apply_overrides(theta: Mapping[str, np.ndarray], overrides: Mapping[str, dict]) -> dict:
    """Scenario overrides: {param: {op: set|multiply, value: x}}."""
    out = {k: np.array(v, dtype=float, copy=True) for k, v in theta.items()}
    for p, spec in (overrides or {}).items():
        out[p] = apply_effect(out[p], spec["op"], np.asarray(spec["value"], dtype=float))
    return out
