import numpy as np

from formcheck.rules import RepTracker, load_rules
from formcheck.skeleton import ANGLE_NAMES

FPS = 30.0
FRONT, SIDE = 0.8, 0.2


def run(exercise: str, signals: dict[str, np.ndarray], view: float):
    """Feed per-frame angle signals; unspecified angles are NaN. Returns (tracker, all cue texts)."""
    tracker = RepTracker(exercise, load_rules())
    n = len(next(iter(signals.values())))
    texts = []
    for i in range(n):
        frame = {name: float("nan") for name in (*ANGLE_NAMES, "torso_lean")}
        frame.update({k: float(v[i]) for k, v in signals.items()})
        texts += [c.text for c in tracker.update(frame, view, i / FPS)]
    return tracker, texts


def reps(top: float, bottom: float, n_reps: int = 3, seconds: float = 2.0) -> np.ndarray:
    """``n_reps`` cosine reps going top → bottom → top, with a short rest before and after."""
    per = int(seconds * FPS)
    phase = np.linspace(0, 2 * np.pi, per, endpoint=False)
    rep = bottom + (top - bottom) * (0.5 + 0.5 * np.cos(phase))
    # Rest after is longer than the tracker's tail, so the last rep is judged.
    return np.concatenate([np.full(int(FPS / 2), top), *([rep] * n_reps), np.full(int(FPS), top)])


def test_good_squats_counted_without_cues():
    knee = reps(175, 85)
    tracker, texts = run("squat", {"left_knee": knee, "right_knee": knee, "torso_lean": knee * 0 + 20}, SIDE)
    assert tracker.reps == 3
    assert texts == []


def test_shallow_squat_go_deeper():
    knee = reps(175, 125)
    tracker, texts = run("squat", {"left_knee": knee, "right_knee": knee}, SIDE)
    assert tracker.reps == 3
    assert texts.count("Go deeper") == 3


def test_depth_not_judged_from_the_front():
    knee = reps(175, 125)
    _, texts = run("squat", {"left_knee": knee, "right_knee": knee}, FRONT)
    assert "Go deeper" not in texts


def test_asymmetric_jumping_jacks():
    left, right = reps(20, 120, seconds=1.0), reps(20, 170, seconds=1.0)  # rest low, arms up
    tracker, texts = run("jumping_jacks", {"left_shoulder": left, "right_shoulder": right}, FRONT)
    assert tracker.reps == 3
    assert "Lift your left arm further" in texts
    assert "Lift your right arm further" not in texts


def test_symmetry_skipped_side_on():
    left, right = reps(20, 120, seconds=1.0), reps(20, 170, seconds=1.0)
    _, texts = run("jumping_jacks", {"left_shoulder": left, "right_shoulder": right}, SIDE)
    assert not any("Lift your" in t for t in texts)


def test_pushup_hip_sag():
    elbow = reps(170, 80)
    body = np.where(elbow < 120, 140.0, 175.0)  # hips drop at the bottom
    _, texts = run(
        "pushup", {"left_elbow": elbow, "right_elbow": elbow, "left_body": body, "right_body": body}, SIDE
    )
    assert "Keep your body in a straight line from shoulders to ankles" in texts


def test_jitter_is_not_a_rep():
    knee = 170 + np.random.default_rng(0).normal(0, 4, 300)
    tracker, _ = run("squat", {"left_knee": knee, "right_knee": knee}, SIDE)
    assert tracker.reps == 0


def test_missing_joints_do_not_crash():
    knee = reps(175, 85)
    left = knee.copy()
    left[::3] = np.nan
    tracker, _ = run("squat", {"left_knee": left, "right_knee": np.full_like(knee, np.nan)}, FRONT)
    assert tracker.reps == 3


def test_lockout_after_exit_counts_as_part_of_the_rep():
    # Starts at the top of a pull-up (bent arms) and lowers to a full hang; the 170° lockout
    # happens after the exit crossing (140°), which the tail must capture.
    elbow = np.concatenate([np.linspace(60, 170, 40), np.full(30, 170.0)])
    tracker, texts = run("pullup", {"left_elbow": elbow, "right_elbow": elbow}, FRONT)
    assert tracker.reps == 1
    assert "Lower all the way to straight arms" not in texts


def test_clean_and_jerk_counts_on_hold_overhead():
    shoulder = np.concatenate([np.full(15, 20.0), np.linspace(20, 165, 30), np.full(30, 165.0)])
    elbow = np.where(shoulder > 130, 150.0, 90.0)  # arms not locked out overhead
    tracker, texts = run(
        "clean_and_jerk",
        {"left_shoulder": shoulder, "right_shoulder": shoulder, "left_elbow": elbow, "right_elbow": elbow},
        FRONT,
    )
    assert tracker.reps == 1
    assert "Lock out your arms overhead" in texts


def test_flush_judges_a_rep_cut_off_by_the_end_of_the_video():
    knee = reps(175, 125, n_reps=1)[: int(2.6 * FPS)]  # ends just after the rep, mid-tail
    tracker, texts = run("squat", {"left_knee": knee, "right_knee": knee}, SIDE)
    assert tracker.reps == 1 and texts == []
    assert [c.text for c in tracker.flush()] == ["Go deeper"]
