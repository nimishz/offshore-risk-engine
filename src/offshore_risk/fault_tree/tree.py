"""Fault tree representation and evaluation.

Why more than one evaluation method?
------------------------------------
Multiplying gate probabilities bottom-up is exact only if every basic event
appears once in the tree. Real protection systems share support systems (a
power supply feeding two pumps, a common-cause event) and those shared events
appear under several gates. Gate multiplication then silently assumes the two
branches are independent when they are not.

Methods provided:

``independent``  gate-by-gate arithmetic; exact only without repeated events.
``exact``        Shannon decomposition on repeated basic events, then gate
                 arithmetic; exact for independent basic events. Cost 2^r.
``rare_event``   sum over minimal cut sets of product of probabilities (upper
                 approximation, accurate when probabilities are small).
``mcub``         min-cut upper bound 1 - prod(1 - P(C)).
``enumeration``  brute force over all 2^n basic-event states (validation only).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BasicEvent:
    name: str
    description: str = ""


@dataclass(frozen=True)
class Gate:
    name: str
    kind: str  # "AND" | "OR" | "KOFN"
    inputs: tuple
    k: int | None = None
    description: str = ""

    def __post_init__(self):
        if self.kind not in ("AND", "OR", "KOFN"):
            raise ValueError(f"unknown gate type {self.kind}")
        if self.kind == "KOFN" and not (1 <= (self.k or 0) <= len(self.inputs)):
            raise ValueError("KOFN gate needs 1 <= k <= number of inputs")
        if not self.inputs:
            raise ValueError("gate needs at least one input")


Node = BasicEvent | Gate


def AND(name: str, *inputs: Node, description: str = "") -> Gate:
    return Gate(name, "AND", tuple(inputs), description=description)


def OR(name: str, *inputs: Node, description: str = "") -> Gate:
    return Gate(name, "OR", tuple(inputs), description=description)


def KOFN(name: str, k: int, *inputs: Node, description: str = "") -> Gate:
    return Gate(name, "KOFN", tuple(inputs), k=k, description=description)


def _at_least_k(k: int, probs: list):
    """P(at least k of the independent inputs occur)."""
    shape = np.broadcast(*[np.asarray(p) for p in probs]).shape
    dist = [np.ones(shape)] + [np.zeros(shape) for _ in probs]
    for p in probs:
        p = np.asarray(p, dtype=float)
        for j in range(len(probs), 0, -1):
            dist[j] = dist[j] * (1 - p) + dist[j - 1] * p
        dist[0] = dist[0] * (1 - p)
    return sum(dist[k:])


class FaultTree:
    def __init__(self, top: Gate, name: str | None = None):
        self.top = top
        self.name = name or top.name

    # --------------------------------------------------------- structure
    def _walk(self, node: Node):
        yield node
        if isinstance(node, Gate):
            for child in node.inputs:
                yield from self._walk(child)

    @property
    def basic_events(self) -> list[str]:
        return sorted({n.name for n in self._walk(self.top) if isinstance(n, BasicEvent)})

    @property
    def gates(self) -> list[str]:
        seen = []
        for n in self._walk(self.top):
            if isinstance(n, Gate) and n.name not in seen:
                seen.append(n.name)
        return seen

    @property
    def repeated_events(self) -> list[str]:
        """Basic events that occur under more than one path to the top."""
        c = Counter(n.name for n in self._walk(self.top) if isinstance(n, BasicEvent))
        return sorted(k for k, v in c.items() if v > 1)

    # --------------------------------------------------------- evaluation
    def _gate_prob(self, node: Node, p: Mapping):
        if isinstance(node, BasicEvent):
            return np.asarray(p[node.name], dtype=float)
        vals = [self._gate_prob(c, p) for c in node.inputs]
        if node.kind == "AND":
            out = 1.0
            for v in vals:
                out = out * v
            return np.asarray(out)
        if node.kind == "OR":
            q = 1.0
            for v in vals:
                q = q * (1 - v)
            return 1 - np.asarray(q)
        return _at_least_k(node.k, vals)

    def _exact(self, p: Mapping, repeated: list[str]):
        if not repeated:
            return self._gate_prob(self.top, p)
        x, rest = repeated[0], repeated[1:]
        px = np.asarray(p[x], dtype=float)
        p1 = dict(p)
        p1[x] = 1.0
        p0 = dict(p)
        p0[x] = 0.0
        return px * self._exact(p1, rest) + (1 - px) * self._exact(p0, rest)

    def probability(self, probs: Mapping, method: str = "exact"):
        missing = set(self.basic_events) - set(probs)
        if missing:
            raise KeyError(f"missing basic-event probabilities: {sorted(missing)}")
        if method == "independent":
            return self._gate_prob(self.top, probs)
        if method == "exact":
            return self._exact(probs, self.repeated_events)
        if method in ("rare_event", "mcub"):
            terms = []
            for cs in self.minimal_cut_sets():
                t = 1.0
                for e in sorted(cs):  # fixed order -> bit-identical results across runs
                    t = t * np.asarray(probs[e], dtype=float)
                terms.append(t)
            if method == "rare_event":
                return sum(terms)
            q = 1.0
            for t in terms:
                q = q * (1 - t)
            return 1 - q
        if method == "enumeration":
            return self._enumerate(probs)
        raise ValueError(f"unknown method {method}")

    # structure function for enumeration
    def _occurs(self, node: Node, state: Mapping[str, bool]) -> bool:
        if isinstance(node, BasicEvent):
            return state[node.name]
        vals = [self._occurs(c, state) for c in node.inputs]
        if node.kind == "AND":
            return all(vals)
        if node.kind == "OR":
            return any(vals)
        return sum(vals) >= node.k

    def _enumerate(self, probs: Mapping):
        events = self.basic_events
        if len(events) > 22:
            raise ValueError("enumeration limited to <= 22 basic events")
        total = 0.0
        for bits in product((False, True), repeat=len(events)):
            state = dict(zip(events, bits))
            if self._occurs(self.top, state):
                w = 1.0
                for e, b in state.items():
                    pe = np.asarray(probs[e], dtype=float)
                    w = w * (pe if b else 1 - pe)
                total = total + w
        return total

    # --------------------------------------------------------- cut sets
    def minimal_cut_sets(self) -> list[frozenset]:
        """MOCUS top-down expansion followed by removal of non-minimal sets."""

        def expand(node: Node) -> list[frozenset]:
            if isinstance(node, BasicEvent):
                return [frozenset([node.name])]
            child_sets = [expand(c) for c in node.inputs]
            if node.kind == "OR":
                return [s for cs in child_sets for s in cs]
            if node.kind == "AND":
                return _and_combine(child_sets)
            out = []
            for combo in combinations(child_sets, node.k):
                out.extend(_and_combine(list(combo)))
            return out

        sets = set(expand(self.top))
        minimal = [s for s in sets if not any(o < s for o in sets)]
        return sorted(minimal, key=lambda s: (len(s), sorted(s)))

    # --------------------------------------------------------- importance
    def importance(self, probs: Mapping) -> pd.DataFrame:
        """Birnbaum, Fussell-Vesely, RAW and RRW for each basic event (exact method).

        Arrays of probabilities are reduced to their mean before evaluation.
        """
        p = {k: float(np.mean(v)) for k, v in probs.items()}
        top = float(self.probability(p))
        rows = []
        for e in self.basic_events:
            p1, p0 = dict(p), dict(p)
            p1[e], p0[e] = 1.0, 0.0
            t1, t0 = float(self.probability(p1)), float(self.probability(p0))
            rows.append(
                {
                    "basic_event": e,
                    "probability": p[e],
                    "birnbaum": t1 - t0,
                    "fussell_vesely": (top - t0) / top if top > 0 else np.nan,
                    "raw": t1 / top if top > 0 else np.nan,
                    "rrw": top / t0 if t0 > 0 else np.inf,
                }
            )
        return pd.DataFrame(rows).sort_values("fussell_vesely", ascending=False).reset_index(drop=True)

    # --------------------------------------------------------- display
    def to_text(self) -> str:
        lines: list[str] = []
        rep = set(self.repeated_events)

        def rec(node: Node, prefix: str, last: bool):
            conn = "└── " if last else "├── "
            if isinstance(node, BasicEvent):
                tag = "  (repeated)" if node.name in rep else ""
                lines.append(f"{prefix}{conn}{node.name}{tag}")
                return
            label = f"{node.kind}" + (f" {node.k}/{len(node.inputs)}" if node.kind == "KOFN" else "")
            lines.append(f"{prefix}{conn}{node.name} [{label}]")
            ext = "    " if last else "│   "
            for i, c in enumerate(node.inputs):
                rec(c, prefix + ext, i == len(node.inputs) - 1)

        lines.append(f"{self.top.name} [{self.top.kind}]")
        for i, c in enumerate(self.top.inputs):
            rec(c, "", i == len(self.top.inputs) - 1)
        return "\n".join(lines)


def _and_combine(child_sets: Iterable[list[frozenset]]) -> list[frozenset]:
    out = [frozenset()]
    for cs in child_sets:
        out = [a | b for a in out for b in cs]
    return out
