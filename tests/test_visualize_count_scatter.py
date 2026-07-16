from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pausenet.visualize import _log10_window_density, plot_count_scatter


def test_log10_window_density_assigns_each_point_its_bin_count() -> None:
    x_values = np.array([0.0, 0.0, 0.0, 1.0])
    y_values = np.array([0.0, 0.0, 0.0, 1.0])

    density = _log10_window_density(x_values, y_values, bins=2)

    np.testing.assert_allclose(density[:3], np.log10(3.0))
    np.testing.assert_allclose(density[3], 0.0)


def test_plot_count_scatter_writes_pdf(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    predictions = pd.DataFrame(
        {
            "observed_counts": [0.0, 1.0, 4.0, 9.0, 20.0],
            "predicted_log1p_counts": [0.1, 0.5, 1.4, 2.0, 2.8],
        }
    )
    output_path = tmp_path / "count_scatter.pdf"

    plot_count_scatter(predictions, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
