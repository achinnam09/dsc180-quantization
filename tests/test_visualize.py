import numpy as np

from quantization.visualize import plot_error_comparison, plot_magnitudes


def test_plot_magnitudes_writes_file(tmp_path):
    path = tmp_path / "magnitudes.png"

    plot_magnitudes(np.random.randn(16, 8), np.random.randn(16, 8), path)

    assert path.stat().st_size > 0


def test_plot_error_comparison_writes_file(tmp_path):
    errors = {
        "key_reconstruction_error": 1.0,
        "attention_score_error": 2.0,
        "value_reconstruction_error": 3.0,
        "attention_output_error": 4.0,
    }
    path = tmp_path / "errors.png"

    plot_error_comparison({"token": errors, "channel": errors}, path)

    assert path.stat().st_size > 0
