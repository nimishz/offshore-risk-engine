"""Event tree for a hydrocarbon release.

Branch parameters (keys in the parameter mapping):

``p_detect``   gas detection triggers ESD (coverage x hardware)
``p_isolate``  ESD isolates the inventory given detection
``p_ign_iso``  ignition probability of an isolated release
``p_ign``      ignition probability of an unisolated release
``p_exp``      explosion given ignition
``p_fp_ok``    fire protection (firewater + deluge + activation) works

Simplifications (documented in docs/limitations.md): an undetected release is
treated as unisolated even though flame detection might later trigger ESD;
fire-protection success does not depend on whether the fire is jet or pool.
"""

from __future__ import annotations

from .tree import EventTree, Split

OUTCOMES = [
    "controlled_release",
    "uncontrolled_release",
    "minor_fire",
    "major_fire",
    "explosion",
    "catastrophic",
]
FIRE_OUTCOMES = ["minor_fire", "major_fire", "explosion", "catastrophic"]


def _after_release(p_ign: str, no_ignition: str, fire_ok: str, fire_fail: str) -> Split:
    """Ignition -> explosion -> fire-protection sub-tree shared by both isolation states."""
    return Split(
        "Ignition",
        p_ign,
        yes=Split(
            "Explosion",
            "p_exp",
            yes=Split("Fire protection works", "p_fp_ok", yes="explosion", no="catastrophic"),
            no=Split("Fire protection works", "p_fp_ok", yes=fire_ok, no=fire_fail),
        ),
        no=no_ignition,
    )


def build_release_tree() -> EventTree:
    isolated = _after_release("p_ign_iso", "controlled_release", "minor_fire", "major_fire")
    unisolated = _after_release("p_ign", "uncontrolled_release", "major_fire", "catastrophic")
    root = Split(
        "Detected",
        "p_detect",
        yes=Split("Isolated", "p_isolate", yes=isolated, no=unisolated),
        no=unisolated,
    )
    return EventTree(root, OUTCOMES, name="Hydrocarbon release")


RELEASE_TREE = build_release_tree()
