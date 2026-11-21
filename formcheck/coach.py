"""Live coach: pose stream → current exercise, rep count and form cues."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from formcheck import skeleton
from formcheck.rules import DEFAULT_RULES, Cue, RepTracker, load_rules

HISTORY_SECONDS = 4.0


@dataclass
class Feedback:
    exercise: str | None = None  # None until the classifier is confident
    confidence: float = 0.0
    reps: int = 0
    cues: list[Cue] = field(default_factory=list)

    @property
    def flagged_angles(self) -> set[str]:
        return {a for c in self.cues for a in c.angles}


class Coach:
    """Classifies the exercise every ``classify_every`` frames (EMA-smoothed probabilities,
    switching only after ``switch_after`` agreeing classifications) and feeds each frame's
    angles to that exercise's rep tracker."""

    def __init__(
        self,
        classifier,  # formcheck.classifier.ExerciseClassifier (or anything with .probabilities/.classes/.cfg)
        rules_path: str | Path = DEFAULT_RULES,
        classify_every: int = 8,
        ema: float = 0.4,
        switch_after: int = 3,
    ):
        self.classifier = classifier
        self.rules = load_rules(rules_path)
        self.classify_every, self.ema, self.switch_after = classify_every, ema, switch_after
        self.threshold = classifier.cfg.score_threshold
        self.reset()

    def reset(self) -> None:
        self.kps: deque[np.ndarray] = deque()
        self.scores: deque[np.ndarray] = deque()
        self.times: deque[float] = deque()
        self.probs: np.ndarray | None = None
        self.exercise: str | None = None
        self.candidate: str | None = None
        self.agree = 0
        self.frame = 0
        self.tracker: RepTracker | None = None
        self.cues: list[Cue] = []

    def _add_cues(self, new: list[Cue], t: float) -> None:
        hold = self.rules.get("hold_seconds", 3.0)
        for c in new:
            c.until = t + hold
        texts = {c.text for c in new}
        self.cues = [c for c in self.cues if c.text not in texts] + new

    def _restart(self, t: float) -> None:
        """Time went backwards (video looped): judge a rep still in progress, start over, and
        keep the cues on screen for the rest of their hold time."""
        last = self.times[-1]
        if self.tracker is not None:
            self._add_cues(self.tracker.flush(), last)
        cues = [c for c in self.cues if c.until > last]
        for c in cues:
            c.until = t + (c.until - last)
        self.reset()
        self.cues = cues

    def _classify(self) -> None:
        p = self.classifier.probabilities(np.stack(self.kps), np.stack(self.scores), np.array(self.times))
        if p is None:
            return
        self.probs = p if self.probs is None else self.ema * p + (1 - self.ema) * self.probs
        best = int(self.probs.argmax())
        name = (
            self.classifier.classes[best] if self.probs[best] >= self.classifier.cfg.unknown_below else None
        )
        if name == self.exercise:
            self.candidate, self.agree = None, 0
            return
        self.agree = self.agree + 1 if name == self.candidate else 1
        self.candidate = name
        if self.agree >= self.switch_after:
            self.exercise, self.candidate, self.agree = name, None, 0
            has_rules = name is not None and name in self.rules["exercises"]
            self.tracker = RepTracker(name, self.rules) if has_rules else None
            self.cues = []

    def update(self, keypoints: np.ndarray, scores: np.ndarray, t: float) -> Feedback:
        """Feed one frame's pose (``(17, 2)`` pixels, ``(17,)`` scores) at time ``t`` seconds."""
        if self.times and t <= self.times[-1]:  # video looped or restarted
            self._restart(t)
        self.kps.append(np.asarray(keypoints, np.float32))
        self.scores.append(np.asarray(scores, np.float32))
        self.times.append(float(t))
        while t - self.times[0] > HISTORY_SECONDS:
            self.kps.popleft()
            self.scores.popleft()
            self.times.popleft()

        self.frame += 1
        if self.frame % self.classify_every == 0:
            self._classify()

        if self.tracker is not None:
            k, s = self.kps[-1][None], self.scores[-1][None]
            frame_angles = {n: float(v[0]) for n, v in skeleton.angles(k, s, self.threshold).items()}
            recent = max(1, int(len(self.times) / 4))  # ~1 s of history for the view estimate
            ratio = skeleton.view_ratio(
                np.stack(list(self.kps)[-recent:]), np.stack(list(self.scores)[-recent:])
            )
            self._add_cues(self.tracker.update(frame_angles, ratio, t), t)
        self.cues = [c for c in self.cues if c.until > t]

        conf = float(self.probs.max()) if self.probs is not None else 0.0
        return Feedback(self.exercise, conf, self.tracker.reps if self.tracker else 0, list(self.cues))
