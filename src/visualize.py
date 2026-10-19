import matplotlib

matplotlib.use("Agg")  # Write image files without needing a display.
import matplotlib.pyplot as plt
import numpy as np

METHOD_COLORS = {"token": "#2a78d6", "channel": "#eb6834"}
METHOD_LABELS = {"token": "Per-token", "channel": "Per-channel"}

# Panels follow KIVI Table 2: key metrics depend only on how K is quantized,
# value metrics only on how V is quantized.
PANELS = [
    ("Key cache", [
        ("key_reconstruction_error", "Key\nreconstruction"),
        ("attention_score_error", "Attention\nscore"),
    ]),
    ("Value cache", [
        ("value_reconstruction_error", "Value\nreconstruction"),
        ("attention_output_error", "Attention\noutput"),
    ]),
]


def plot_magnitudes(K, V, path):
    """3D surface of |K| and |V| over tokens and channels, like KIVI Figure 2."""
    fig = plt.figure(figsize=(12, 5))
    for i, (name, X) in enumerate([("Key cache", K), ("Value cache", V)]):
        X = np.abs(np.asarray(X, dtype=np.float64))
        tokens, channels = np.meshgrid(
            np.arange(X.shape[0]), np.arange(X.shape[1]), indexing="ij"
        )
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.plot_surface(tokens, channels, X, cmap="Blues", linewidth=0,
                        rstride=4, cstride=1, antialiased=False)
        ax.set_title(name)
        ax.set_xlabel("Token")
        ax.set_ylabel("Channel")
        ax.set_zlabel("Absolute value")
        # Look across the channel axis so outlier channels show as separate
        # ridges instead of overlapping walls that hide the low values behind them.
        ax.view_init(elev=25, azim=-15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_comparison(errors_by_method, path):
    """Grouped bars of relative error per quantization method, like KIVI Table 2.

    errors_by_method maps a method name ('token' or 'channel') to the dict
    returned by compute_error_metrics.
    """
    methods = [m for m in METHOD_COLORS if m in errors_by_method]
    width = 0.8 / len(methods)

    fig, axes = plt.subplots(1, len(PANELS), figsize=(10, 4.5), sharey=True)
    for ax, (title, metrics) in zip(axes, PANELS):
        x = np.arange(len(metrics))
        for j, method in enumerate(methods):
            values = [errors_by_method[method][key] for key, _ in metrics]
            offset = (j - (len(methods) - 1) / 2) * width
            bars = ax.bar(x + offset, values, width * 0.95,
                          color=METHOD_COLORS[method], label=METHOD_LABELS[method])
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, color="#444444")
        ax.set_title(title)
        ax.set_xticks(x, [label for _, label in metrics])
        ax.set_yscale("log")
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("Relative error (log scale)")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
