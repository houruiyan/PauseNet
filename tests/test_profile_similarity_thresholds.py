from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pausenet.metrics import profile_similarity_by_count_threshold
from pausenet.visualize import plot_profile_similarity


def test_profile_similarity_count_thresholds_filter_observed_totals() -> None:
    observed = np.zeros((4, 20), dtype=np.float64)
    for index in range(4):
        observed[index, 2 + 2 * index] = 10

    rows = pd.DataFrame(
        profile_similarity_by_count_threshold(
            observed,
            observed.copy(),
            np.ones(4, dtype=bool),
            observed_counts=np.array([50, 150, 250, 600]),
            resolutions=(1, 5),
            count_thresholds=(0, 100, 200, 500),
            seed=7,
        )
    )

    pausenet = rows.loc[rows["comparison"] == "PauseNet"]
    observed_n = (
        pausenet.groupby("count_threshold", sort=True)["n"].first().to_dict()
    )
    assert observed_n == {0: 4, 100: 3, 200: 2, 500: 1}
    assert np.allclose(pausenet["similarity_1_minus_jsd"], 1.0)


def test_threshold_profile_similarity_plot_is_written(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    rows = []
    for threshold, n in ((0, 100), (100, 50), (200, 20), (500, 5)):
        for comparison, offset in (
            ("Pseudoreplicates", 0.3),
            ("PauseNet", 0.15),
            ("Random profile", 0.0),
        ):
            for resolution, gain in ((1, 0.0), (5, 0.1), (10, 0.15), (20, 0.2)):
                rows.append(
                    {
                        "count_threshold": threshold,
                        "comparison": comparison,
                        "resolution_bp": resolution,
                        "similarity_1_minus_jsd": min(offset + gain, 1.0),
                        "n": n,
                    }
                )

    output_path = tmp_path / "profile_similarity.pdf"
    plot_profile_similarity(pd.DataFrame(rows), output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
