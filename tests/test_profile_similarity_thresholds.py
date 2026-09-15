import numpy as np
import pandas as pd

from pausenet.metrics import profile_similarity_by_count_threshold


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
