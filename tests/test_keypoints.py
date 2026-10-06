from posenet.keypoints import FLIP_INDEX, KEYPOINT_NAMES, NUM_KEYPOINTS, OKS_SIGMAS, SKELETON


def test_flip_index_is_involution():
    assert len(FLIP_INDEX) == NUM_KEYPOINTS
    assert all(FLIP_INDEX[FLIP_INDEX[i]] == i for i in range(NUM_KEYPOINTS))


def test_flip_index_swaps_left_and_right():
    for i, j in enumerate(FLIP_INDEX):
        a, b = KEYPOINT_NAMES[i], KEYPOINT_NAMES[j]
        if a.startswith("left_"):
            assert b == "right_" + a[len("left_") :]
        elif a.startswith("right_"):
            assert b == "left_" + a[len("right_") :]
        else:
            assert i == j


def test_schema_sizes():
    assert NUM_KEYPOINTS == 17
    assert OKS_SIGMAS.shape == (17,)
    assert all(0 <= a < 17 and 0 <= b < 17 for a, b in SKELETON)
