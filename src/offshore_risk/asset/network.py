"""Platform interdependency model.

Two different relationships between platforms are kept separate:

* **Functional dependency** (``dependencies`` in asset_config.yaml): platform X
  needs a service from platform Y (power, well fluids, export route, manning).
  Loss of Y reduces X's capability even if X is physically intact.
* **Physical adjacency** (``bridges``): a fire or explosion on X can escalate to
  a bridge-linked neighbour.

Production capacity for every combination of 'down' platforms (2^6 = 64) is
pre-computed so the simulation can look it up with a bit mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from graphlib import TopologicalSorter

import numpy as np
import pandas as pd


@dataclass
class AssetModel:
    codes: list[str]
    names: dict[str, str]
    production_bopd: np.ndarray  # nominal, per platform
    replacement_value: np.ndarray  # USD, per platform
    release_factor: np.ndarray  # relative leak-source count
    gas_fraction: np.ndarray
    p_power_loss_given_fire: np.ndarray
    dependencies: list[dict]
    bridges: list[tuple[str, str]]
    protection: dict
    pipeline_km: float

    @classmethod
    def from_config(cls, asset_cfg: dict) -> AssetModel:
        plats = asset_cfg["platforms"]
        codes = list(plats)

        def g(key: str) -> np.ndarray:
            return np.array([float(plats[c].get(key, 0.0)) for c in codes])

        return cls(
            codes=codes,
            names={c: plats[c]["name"] for c in codes},
            production_bopd=g("production_bopd"),
            replacement_value=g("replacement_value_usd"),
            release_factor=g("release_equipment_factor"),
            gas_fraction=g("gas_release_fraction"),
            p_power_loss_given_fire=g("p_power_loss_given_fire"),
            dependencies=list(asset_cfg.get("dependencies", [])),
            bridges=[tuple(b) for b in asset_cfg.get("bridges", [])],
            protection=dict(asset_cfg.get("protection", {})),
            pipeline_km=float(asset_cfg.get("pipeline", {}).get("export_length_km", 0.0)),
        )

    # ------------------------------------------------------------ helpers
    @property
    def n(self) -> int:
        return len(self.codes)

    def index(self, code: str) -> int:
        return self.codes.index(code)

    @property
    def process_mask(self) -> np.ndarray:
        """Platforms with hydrocarbon process inventory (they need a restart after a shutdown)."""
        return self.release_factor > 0

    @property
    def producer_mask(self) -> np.ndarray:
        return self.production_bopd > 0

    @property
    def total_production(self) -> float:
        return float(self.production_bopd.sum())

    def neighbours(self, code: str) -> list[str]:
        out = []
        for a, b in self.bridges:
            if a == code:
                out.append(b)
            elif b == code:
                out.append(a)
        return out

    def neighbour_matrix(self, max_neighbours: int | None = None) -> np.ndarray:
        """(n, max_neighbours) int array of neighbour indices, padded with -1."""
        nb = [[self.index(x) for x in self.neighbours(c)] for c in self.codes]
        needed = max([len(x) for x in nb] + [1])
        m = max_neighbours or needed
        if m < needed:
            raise ValueError(f"a platform has {needed} bridge neighbours; max_neighbours={m} is too small")
        out = -np.ones((self.n, m), dtype=int)
        for i, row in enumerate(nb):
            out[i, : len(row)] = row
        return out

    def topological_order(self) -> list[str]:
        ts = TopologicalSorter({c: set() for c in self.codes})
        for d in self.dependencies:
            ts.add(d["platform"], d["provider"])
        return list(ts.static_order())  # raises CycleError if dependencies are circular

    # ------------------------------------------------------------ capacity
    def availability(self, down: set[str]) -> dict[str, float]:
        """Capability fraction of each platform given a set of platforms that are down."""
        a: dict[str, float] = {}
        for c in self.topological_order():
            v = 0.0 if c in down else 1.0
            for d in self.dependencies:
                if d["platform"] == c:
                    v *= 1.0 - float(d["impact"]) * (1.0 - a[d["provider"]])
            a[c] = v
        return a

    def capacity(self, down: set[str]) -> float:
        a = self.availability(down)
        return float(sum(self.production_bopd[i] * a[c] for i, c in enumerate(self.codes)))

    def capacity_table(self) -> np.ndarray:
        """Capacity (bbl/d) for every down-set, indexed by bit mask (bit i = platform i down)."""
        out = np.zeros(2**self.n)
        for mask in range(2**self.n):
            down = {c for i, c in enumerate(self.codes) if mask >> i & 1}
            out[mask] = self.capacity(down)
        return out

    def single_outage_table(self) -> pd.DataFrame:
        """Production lost when each platform alone is down (shows dependency effects)."""
        rows = []
        q = self.total_production
        for c in self.codes:
            a = self.availability({c})
            rows.append(
                {
                    "platform": c,
                    "name": self.names[c],
                    "own_production_bopd": float(self.production_bopd[self.index(c)]),
                    "complex_production_lost_bopd": q - self.capacity({c}),
                    "fraction_of_complex": (q - self.capacity({c})) / q,
                    "platforms_affected": ", ".join(k for k, v in a.items() if v < 1.0 and k != c),
                }
            )
        return pd.DataFrame(rows)


def lost_production_bbl(downtime_days: np.ndarray, cap_table: np.ndarray, rate_factor=1.0) -> np.ndarray:
    """Barrels lost for outages where platform i is down for downtime_days[:, i] from t = 0.

    Piecewise-constant integration: between consecutive sorted durations the set
    of down platforms is fixed, so capacity is a table lookup.
    """
    d = np.asarray(downtime_days, dtype=float)
    if d.ndim == 1:
        d = d[None, :]
    n_events, n_plat = d.shape
    full = cap_table[0]
    s = np.sort(d, axis=1)
    weights = 1 << np.arange(n_plat)
    lost = np.zeros(n_events)
    t_prev = np.zeros(n_events)
    for k in range(n_plat):
        t_k = s[:, k]
        dt = t_k - t_prev
        mask = ((d >= t_k[:, None]) & (d > 0)) @ weights
        lost += (full - cap_table[mask]) * dt
        t_prev = t_k
    return lost * rate_factor
