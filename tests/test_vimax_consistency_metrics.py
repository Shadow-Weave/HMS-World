import numpy as np
import pytest

from evals.vimax.score import consistency_metrics


def test_scores_use_distinct_pairs_and_official_scene_groups():
    metrics, similarities = consistency_metrics(np.array([[2, 0], [1, 0], [0, 3]]), [1, 1, 2])
    assert metrics["IC"] == pytest.approx(1)
    assert metrics["CC"] == pytest.approx(0)
    assert metrics["GC"] == pytest.approx(1 / 3)
    assert metrics["intra_scene_pairs"] == 1
    assert metrics["cross_scene_pairs"] == 2
    assert metrics["all_pairs"] == 3
    assert similarities.shape == (3, 3)


def test_thirteen_shot_case_pair_counts():
    metrics, _ = consistency_metrics(np.ones((13, 2)), [1] * 4 + [2] * 4 + [3] * 5)
    assert metrics["intra_scene_pairs"] == 22
    assert metrics["cross_scene_pairs"] == 56
    assert metrics["all_pairs"] == 78
