import numpy as np

from formcheck import skeleton
from formcheck.classifier import ClassifierConfig, ExerciseClassifier, build_model, save
from formcheck.coach import Coach

CLASSES = ["a", "b", "c"]


def test_build_and_round_trip(tmp_path):
    cfg = ClassifierConfig(classes=CLASSES, window=16, width=16, dilations=(1, 2))
    model = build_model(cfg)
    x = np.random.default_rng(0).normal(size=(2, 16, skeleton.NUM_FEATURES)).astype(np.float32)
    p = np.asarray(model(x, training=False))
    assert p.shape == (2, 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)

    save(tmp_path, model, cfg)
    clf = ExerciseClassifier(tmp_path)
    assert clf.classes == CLASSES
    np.testing.assert_allclose(np.asarray(clf.model(x, training=False)), p, atol=1e-5)

    kps = np.random.default_rng(1).uniform(0, 200, size=(40, 17, 2)).astype(np.float32)
    scores = np.ones((40, 17), np.float32)
    probs = clf.probabilities(kps, scores, np.arange(40) / 30.0)
    assert probs.shape == (3,)


class FixedClassifier:
    """Always says squat, so the coach's switching and rep tracking can be tested alone."""

    classes = ["squat", "pushup"]
    cfg = ClassifierConfig(classes=classes, unknown_below=0.5)

    def probabilities(self, kps, scores, times):
        return np.array([0.9, 0.1])


def test_coach_switches_and_resets_on_loop():
    from tests.test_skeleton import standing

    coach = Coach(FixedClassifier(), classify_every=2, switch_after=3)
    kps, scores = standing()
    fb = None
    for i in range(20):
        fb = coach.update(kps[0], scores[0], i / 30.0)
    assert fb.exercise == "squat"
    assert fb.reps == 0
    fb = coach.update(kps[0], scores[0], 0.0)  # video looped
    assert fb.exercise is None
