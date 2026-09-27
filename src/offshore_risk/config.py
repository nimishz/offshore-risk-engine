"""Configuration loading and the epistemic parameter registry."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import yaml

from .distributions import Distribution, make_distribution

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = Path(os.environ.get("OFFSHORE_RISK_CONFIG", REPO_ROOT / "config"))


@dataclass
class Parameter:
    name: str
    spec: dict
    dist: Distribution

    @property
    def unit(self) -> str:
        return self.spec.get("unit", "")


class ParameterRegistry:
    """Ordered collection of epistemic parameters with their distributions.

    ``sample(u)`` maps an ``(n, P)`` array of uniforms to a dict of arrays,
    one per parameter, in registry order. Fixed parameters consume a column
    too, so that adding/removing uncertainty on one parameter never shifts the
    random numbers used for the others.
    """

    def __init__(self, specs: Mapping[str, dict]):
        self.params: dict[str, Parameter] = {
            k: Parameter(k, dict(v), make_distribution(v["distribution"])) for k, v in specs.items()
        }

    # --------------------------------------------------------------- access
    @property
    def names(self) -> list[str]:
        return list(self.params)

    def __len__(self) -> int:
        return len(self.params)

    def __getitem__(self, name: str) -> Parameter:
        return self.params[name]

    def uncertain_names(self) -> list[str]:
        return [k for k, p in self.params.items() if not p.dist.is_fixed]

    # ------------------------------------------------------------- sampling
    def sample(self, u: np.ndarray) -> dict[str, np.ndarray]:
        u = np.atleast_2d(u)
        if u.shape[1] != len(self):
            raise ValueError(f"expected {len(self)} columns of uniforms, got {u.shape[1]}")
        return {k: p.dist.ppf(u[:, j]) for j, (k, p) in enumerate(self.params.items())}

    def at_quantile(self, n: int = 1, q: float = 0.5, overrides: Mapping[str, float] | None = None):
        """All parameters at quantile q (default: medians), optionally overriding some quantiles."""
        overrides = overrides or {}
        return {
            k: np.full(n, p.dist.quantile(overrides.get(k, q))) for k, p in self.params.items()
        }

    def medians(self) -> dict[str, float]:
        return {k: p.dist.median() for k, p in self.params.items()}

    def with_distribution(self, name: str, dist_spec: dict) -> "ParameterRegistry":
        new = copy.deepcopy(self)
        spec = dict(new.params[name].spec)
        spec["distribution"] = dict(dist_spec)
        new.params[name] = Parameter(name, spec, make_distribution(dist_spec))
        return new

    # ------------------------------------------------------------ reporting
    def table(self) -> pd.DataFrame:
        rows = []
        for k, p in self.params.items():
            d = p.dist
            rows.append(
                {
                    "variable": k,
                    "definition": p.spec.get("definition", ""),
                    "unit": p.spec.get("unit", ""),
                    "category": p.spec.get("category", ""),
                    "distribution": d.describe(),
                    "median": d.median(),
                    "p05": d.quantile(0.05),
                    "p95": d.quantile(0.95),
                    "basis": p.spec.get("basis", ""),
                    "uncertainty": p.spec.get("uncertainty", "epistemic"),
                    "rationale": " ".join(str(p.spec.get("rationale", "")).split()),
                    "notes": p.spec.get("notes", ""),
                }
            )
        return pd.DataFrame(rows)


@dataclass
class ModelConfig:
    asset: dict
    registry: ParameterRegistry
    aleatory: dict
    drivers: dict
    financial: dict
    mitigations: dict
    budget_usd: float
    appetite: dict
    scenarios: dict
    source_dir: Path = field(default=DEFAULT_CONFIG_DIR)

    def copy(self) -> "ModelConfig":
        return copy.deepcopy(self)


def _read(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config(config_dir: str | Path | None = None) -> ModelConfig:
    d = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    asset = _read(d / "asset_config.yaml")
    risk = _read(d / "risk_parameters.yaml")
    mit = _read(d / "mitigation_config.yaml")
    app = _read(d / "risk_appetite.yaml")
    scn = _read(d / "scenarios.yaml")
    return ModelConfig(
        asset=asset,
        registry=ParameterRegistry(risk["epistemic"]),
        aleatory=risk["aleatory"],
        drivers=risk["drivers"],
        financial=risk["financial"],
        mitigations=mit["mitigations"],
        budget_usd=float(mit["budget_usd"]),
        appetite=app,
        scenarios=scn["scenarios"],
        source_dir=d,
    )
