"""Generic event tree with array-valued branch probabilities.

A tree is built from :class:`Split` nodes (a yes/no question whose 'yes'
probability is looked up by key in a parameter mapping) and string leaves
(outcome labels). Because probabilities may be numpy arrays, one call
evaluates the tree for every simulated event at once.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

Prob = str | float | Callable[[Mapping], np.ndarray]


@dataclass(frozen=True)
class Split:
    question: str
    p_yes: Prob
    yes: Node
    no: Node


Node = Split | str


@dataclass(frozen=True)
class Leaf:
    index: int
    outcome: str
    path: tuple  # ((question, answer_bool), ...)

    def answered(self, question: str):
        for q, a in self.path:
            if q == question:
                return a
        return None


class EventTree:
    def __init__(self, root: Node, outcome_order: list[str], name: str = "event tree"):
        self.root = root
        self.name = name
        self.outcome_order = list(outcome_order)
        leaves: list[Leaf] = []

        def rec(node, path):
            if isinstance(node, str):
                if node not in self.outcome_order:
                    raise ValueError(f"outcome {node!r} not in outcome_order")
                leaves.append(Leaf(len(leaves), node, tuple(path)))
                return
            rec(node.yes, path + [(node.question, True)])
            rec(node.no, path + [(node.question, False)])

        rec(root, [])
        # Order leaves by outcome severity (then by tree order) so that inverse-CDF
        # sampling maps a larger uniform to an equal-or-worse outcome.
        self.leaves = sorted(leaves, key=lambda lf: (self.outcome_order.index(lf.outcome), lf.index))
        self._leaf_outcome_idx = np.array([self.outcome_order.index(lf.outcome) for lf in self.leaves])

    @staticmethod
    def _p(prob: Prob, params: Mapping):
        if callable(prob):
            return np.asarray(prob(params), dtype=float)
        if isinstance(prob, str):
            return np.asarray(params[prob], dtype=float)
        return np.asarray(prob, dtype=float)

    def leaf_probabilities(self, params: Mapping) -> np.ndarray:
        """Array (..., n_leaves) of path probabilities in severity order."""
        probs: dict[int, np.ndarray] = {}
        counter = [0]

        def rec(node, acc):
            if isinstance(node, str):
                probs[counter[0]] = acc
                counter[0] += 1
                return
            p = np.clip(self._p(node.p_yes, params), 0.0, 1.0)
            rec(node.yes, acc * p)
            rec(node.no, acc * (1.0 - p))

        rec(self.root, np.asarray(1.0))
        shape = np.broadcast(*probs.values()).shape
        return np.stack([np.broadcast_to(probs[lf.index], shape) for lf in self.leaves], axis=-1)

    def outcome_probabilities(self, params: Mapping) -> dict[str, np.ndarray]:
        lp = self.leaf_probabilities(params)
        out = {}
        for j, o in enumerate(self.outcome_order):
            out[o] = lp[..., self._leaf_outcome_idx == j].sum(axis=-1)
        return out

    def sample_leaves(self, params: Mapping, u: np.ndarray) -> np.ndarray:
        """Sample one leaf per event by inverse CDF over severity-ordered leaves."""
        lp = self.leaf_probabilities(params)
        cdf = np.cumsum(lp, axis=-1)
        cdf[..., -1] = 1.0
        return (np.asarray(u)[..., None] > cdf).sum(axis=-1)

    def leaf_outcome_index(self, leaf_ids: np.ndarray) -> np.ndarray:
        return self._leaf_outcome_idx[leaf_ids]

    def leaf_attribute(self, question: str) -> np.ndarray:
        """Boolean array over leaves: was `question` answered yes on the path?"""
        return np.array([bool(lf.answered(question)) for lf in self.leaves])

    def path_table(self, params: Mapping):
        import pandas as pd

        lp = self.leaf_probabilities({k: np.mean(v) for k, v in params.items()})
        rows = []
        for j, lf in enumerate(self.leaves):
            rows.append(
                {
                    "path": " -> ".join(f"{q}={'Y' if a else 'N'}" for q, a in lf.path),
                    "outcome": lf.outcome,
                    "probability": float(lp[j]),
                }
            )
        return pd.DataFrame(rows)
