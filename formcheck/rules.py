"""Rep counting and rule-based form cues from joint-angle time series (thresholds in rules.json).

There is no labelled good/bad form data, so form is judged by per-exercise rules: each rep
is cut out of the primary angle's signal with a hysteresis state machine, then each check
looks at the rep's angles (e.g. "knee never got below 100° → Go deeper", "left shoulder
peaked 25° lower than the right → Lift your left arm further").
"""

from __future__ import annotations

import contextlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_RULES = Path(__file__).with_name("rules.json")
PEAK_FRAMES = 3  # frames either side of the primary extreme used by "at": "peak" checks
LEAD_IN_FRAMES = 6  # rest frames kept before a rep starts
TAIL_SECONDS = 0.8  # longest a rep keeps recording after the exit crossing
TURN_DEGREES = 3.0  # the tail ends once the angle turns back this far from rest


@dataclass
class Cue:
    text: str
    angles: tuple[str, ...]  # angle names to highlight, e.g. ("left_shoulder",)
    until: float = 0.0  # time the cue stops showing


def load_rules(path: str | Path = DEFAULT_RULES) -> dict:
    return json.loads(Path(path).read_text())


def classify_view(ratio: float, view_cfg: dict) -> str:
    """Shoulder width / torso length → "front", "side" or "diagonal" ("unknown" if NaN)."""
    if not np.isfinite(ratio):
        return "unknown"
    if ratio >= view_cfg["front_above"]:
        return "front"
    if ratio <= view_cfg["side_below"]:
        return "side"
    return "diagonal"


def _both(name: str) -> tuple[str, ...]:
    """Angle family → the per-side angle names ("knee" → left_knee, right_knee)."""
    return (name,) if name == "torso_lean" else (f"left_{name}", f"right_{name}")


def _family(rep: dict[str, np.ndarray], name: str) -> np.ndarray:
    """Mean of both sides per frame, using whichever side is known."""
    cols = np.stack([rep[n] for n in _both(name)])
    with _quiet():
        return np.nanmean(cols, axis=0)


@contextlib.contextmanager
def _quiet():
    """Silence numpy's all-NaN slice warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def _enough(x: np.ndarray, frac: float = 0.5) -> bool:
    return bool(x.size > 0 and np.isfinite(x).mean() >= frac)


def run_check(check: dict, rep: dict[str, np.ndarray], peak: slice, view: str) -> Cue | None:
    """One check on one finished rep. ``rep`` maps angle name → (T,) degrees."""
    views = check.get("view", "any")
    views = [views] if isinstance(views, str) else views
    if "any" not in views and view not in views:
        return None
    sel = peak if check.get("at") == "peak" else slice(None)
    kind, value, name = check["type"], float(check["value"]), check["angle"]

    if kind == "symmetry":
        left, right = (rep[n][sel] for n in _both(name))
        if not (_enough(left) and _enough(right)):
            return None
        reduce = np.nanmax if check.get("extreme", "max") == "max" else np.nanmin
        lv, rv = float(reduce(left)), float(reduce(right))
        if abs(lv - rv) <= value:
            return None
        # The lagging side reached a smaller max (or a larger min) than the other.
        lagging = (
            ("left" if lv < rv else "right")
            if check.get("extreme", "max") == "max"
            else ("left" if lv > rv else "right")
        )
        other = "right" if lagging == "left" else "left"
        return Cue(check["cue"].format(side=lagging, other=other), (f"{lagging}_{name}",))

    x = _family(rep, name)[sel]
    if not _enough(x):
        return None
    with _quiet():
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    failed = {
        "min_at_most": lo > value,  # never got low enough (depth)
        "min_at_least": lo < value,  # dropped too low (sagging)
        "max_at_least": hi < value,  # never got high enough (lockout, range)
        "max_at_most": hi > value,  # went too far (lean)
    }[kind]
    return Cue(check["cue"], _both(name)) if failed else None


@dataclass
class RepTracker:
    """Counts reps of one exercise and returns cues for each finished rep.

    States: ``rest`` → (primary crosses ``enter``) → ``active`` → (crosses ``exit``; the rep
    counts) → ``tail`` → (the angle stops moving toward rest, or ``TAIL_SECONDS``) → evaluate.
    The tail keeps recording after the exit crossing so lockout at the top is part of the
    rep. With ``count_on_hold`` (seconds) a rep also counts, and is judged, once the primary
    has stayed past ``enter`` that long (e.g. the bar held overhead in a jerk).
    """

    exercise: str
    rules: dict
    reps: int = 0
    _state: str = "rest"
    _ema: float = float("nan")
    _start: float = 0.0
    _away_since: float | None = None
    _counted: bool = False
    _tail_start: float = 0.0
    _best: float = 0.0  # furthest the tail got toward rest
    _frames: list[dict[str, float]] = field(default_factory=list)
    _views: list[float] = field(default_factory=list)

    @property
    def spec(self) -> dict:
        return self.rules["exercises"][self.exercise]

    def _keep_lead_in(self) -> None:
        self._frames, self._views = self._frames[-LEAD_IN_FRAMES:], self._views[-LEAD_IN_FRAMES:]

    def update(self, frame_angles: dict[str, float], view_ratio: float, t: float) -> list[Cue]:
        """Feed one frame's angles (NaN = unknown). Returns cues when a rep has just been judged."""
        spec = self.spec
        primary = _family({k: np.array([v]) for k, v in frame_angles.items()}, spec["primary"])[0]
        if np.isfinite(primary):
            a = self.rules.get("smoothing", 0.4)
            self._ema = primary if not np.isfinite(self._ema) else a * primary + (1 - a) * self._ema
        if not np.isfinite(self._ema):
            return []
        high = spec["rest"] == "high"
        ema = self._ema
        away = ema < spec["enter"] if high else ema > spec["enter"]
        back = ema > spec["exit"] if high else ema < spec["exit"]
        self._away_since = (self._away_since if self._away_since is not None else t) if away else None
        self._frames.append(frame_angles)
        self._views.append(view_ratio)

        cues: list[Cue] = []
        if self._state == "tail":
            self._best = max(self._best, ema) if high else min(self._best, ema)
            turned = (self._best - ema if high else ema - self._best) > TURN_DEGREES
            if not (away or turned or t - self._tail_start > TAIL_SECONDS):
                return []
            cues = self.evaluate(self._frames, self._views)
            self._state = "rest"
            self._keep_lead_in()

        if self._state == "rest":
            if away:
                self._state, self._start, self._counted = "active", t, False
            else:
                self._keep_lead_in()
            return cues

        # active
        if t - self._start > self.rules.get("max_rep_seconds", 10.0):
            self._state = "rest"  # stalled: not a rep
            self._keep_lead_in()
            return cues
        hold = spec.get("count_on_hold")
        if hold and not self._counted and self._away_since is not None and t - self._away_since >= hold:
            self._counted = True
            self.reps += 1
            cues += self.evaluate(self._frames, self._views)
        if back:
            if self._counted or t - self._start < self.rules.get("min_rep_seconds", 0.4):
                self._state = "rest"  # already judged, or too quick to be a rep
                self._keep_lead_in()
            else:
                self.reps += 1
                self._state, self._tail_start, self._best = "tail", t, ema
        return cues

    def flush(self) -> list[Cue]:
        """Judge a rep that is still in its tail (e.g. the video ended right after it)."""
        if self._state != "tail":
            return []
        self._state = "rest"
        cues = self.evaluate(self._frames, self._views)
        self._keep_lead_in()
        return cues

    def evaluate(self, frames: list[dict[str, float]], views: list[float]) -> list[Cue]:
        spec = self.spec
        rep = {k: np.array([f.get(k, np.nan) for f in frames], np.float64) for k in frames[0]}
        with _quiet():
            view = classify_view(float(np.nanmedian(views)) if views else float("nan"), self.rules["view"])
        primary = _family(rep, spec["primary"])
        if np.isfinite(primary).any():
            i = int(np.nanargmin(primary) if spec["rest"] == "high" else np.nanargmax(primary))
        else:
            i = len(primary) // 2
        peak = slice(max(i - PEAK_FRAMES, 0), i + PEAK_FRAMES + 1)
        cues = [run_check(c, rep, peak, view) for c in spec.get("checks", [])]
        return [c for c in cues if c is not None]
