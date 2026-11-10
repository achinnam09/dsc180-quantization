import numpy as np

from quantization.visualize import (
    plot_error_comparison,
    plot_errors_by_layer,
    plot_errors_by_position,
    plot_magnitudes,
)


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


METRICS = ["key_reconstruction_error", "attention_score_error",
           "value_reconstruction_error", "attention_output_error"]


def test_plot_error_comparison_with_std_writes_file(tmp_path):
    errors = dict.fromkeys(METRICS, 1.0)
    std = dict.fromkeys(METRICS, 0.1)
    path = tmp_path / "errors_std.png"

    plot_error_comparison({"token": errors, "channel": errors}, path,
                          std_by_method={"token": std, "channel": std}, title="t")

    assert path.stat().st_size > 0


def _records():
    rng = np.random.default_rng(0)
    return [{"method": m, "layer": lay, "position": pos, **{k: rng.random() for k in METRICS}}
            for m in ("token", "channel") for lay in range(3) for pos in range(250, 260)]


def test_plot_errors_by_layer_writes_file(tmp_path):
    path = tmp_path / "by_layer.png"

    plot_errors_by_layer(_records(), path)

    assert path.stat().st_size > 0


def test_plot_errors_by_position_writes_file(tmp_path):
    path = tmp_path / "by_position.png"

    # Position 255 is a prefix of length 256, a multiple of the residual length.
    plot_errors_by_position(_records(), path, residual_length=128)

    assert path.stat().st_size > 0
