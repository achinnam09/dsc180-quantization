import matplotlib

matplotlib.use("Agg")  # Write image files without needing a display.
import matplotlib.pyplot as plt
import numpy as np

METHOD_COLORS = {"token": "#2a78d6", "channel": "#eb6834", "kivi": "#1baf7a"}
METHOD_LABELS = {"token": "Per-token (K & V)", "channel": "Per-channel (K & V)",
                 "kivi": "KIVI (K per-channel, V per-token)"}
# A second cue besides color, so lines stay distinguishable without it.
METHOD_MARKERS = {"token": "o", "channel": "s", "kivi": "^"}

# Panels follow KIVI Table 2: key metrics depend only on how K is quantized,
# value metrics only on how V is quantized. The last panel quantizes both.
PANELS = [
    ("Key cache", [
        ("key_reconstruction_error", "Key\nreconstruction"),
        ("attention_score_error", "Attention\nscore"),
        ("key_output_error", "Attention\noutput"),
    ]),
    ("Value cache", [
        ("value_reconstruction_error", "Value\nreconstruction"),
        ("attention_output_error", "Attention\noutput"),
    ]),
    ("Both caches", [
        ("combined_output_error", "Attention\noutput"),
    ]),
]

# Titles for the per-metric line plots, which show one metric per panel.
METRIC_TITLES = {
    "key_reconstruction_error": "Key reconstruction",
    "attention_score_error": "Attention score (K quantized)",
    "key_output_error": "Attention output (K quantized)",
    "value_reconstruction_error": "Value reconstruction",
    "attention_output_error": "Attention output (V quantized)",
    "combined_output_error": "Attention output (K + V quantized)",
}


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


def _style_axis(ax):
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_error_comparison(errors_by_method, path, std_by_method=None, title=None):
    """Grouped bars of relative error per quantization method, like KIVI Table 2.

    errors_by_method maps a method name ('token' or 'channel') to the dict
    returned by compute_error_metrics. std_by_method, if given, has the same
    shape and is drawn as error bars.
    """
    methods = [m for m in METHOD_COLORS if m in errors_by_method]
    width = 0.8 / len(methods)
    # Only draw the metrics that were computed, and drop panels left empty.
    computed = errors_by_method[methods[0]]
    panels = [(t, [(k, lbl) for k, lbl in ms if k in computed]) for t, ms in PANELS]
    panels = [(t, ms) for t, ms in panels if ms]

    fig, axes = plt.subplots(1, len(panels), figsize=(4 + 3 * sum(len(ms) for _, ms in panels) / 2, 4.8),
                             sharey=True, width_ratios=[len(ms) for _, ms in panels], squeeze=False)
    axes = axes[0]
    for ax, (panel_title, metrics) in zip(axes, panels):
        x = np.arange(len(metrics))
        for j, method in enumerate(methods):
            values = [errors_by_method[method][key] for key, _ in metrics]
            yerr = None
            if std_by_method is not None:
                std = [std_by_method[method][key] for key, _ in metrics]
                # Errors can't be negative, so clip the lower whisker at zero.
                yerr = [[min(s, v) for s, v in zip(std, values)], std]
            offset = (j - (len(methods) - 1) / 2) * width
            bars = ax.bar(x + offset, values, width * 0.95, yerr=yerr,
                          error_kw={"ecolor": "#888888", "elinewidth": 1, "capsize": 3},
                          color=METHOD_COLORS[method], label=METHOD_LABELS[method])
            if yerr is None:
                ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, color="#444444")
            else:
                # Label above the upper whisker so it doesn't collide with it.
                for xi, v, s in zip(x + offset, values, std):
                    ax.annotate(f"{v:.2f}", (xi, v + s), xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8, color="#444444")
        ax.set_title(panel_title)
        ax.set_xticks(x, [label for _, label in metrics])
        _style_axis(ax)

    axes[0].set_ylabel("Relative error" + (" (mean ± std)" if std_by_method else ""))
    # Legend below the panels so it never covers a bar.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(methods), frameon=False)
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _mean_by(records, key, metric, method):
    """Mean of `metric` for each distinct value of `key`, restricted to one method."""
    groups = {}
    for r in records:
        if r["method"] == method:
            groups.setdefault(r[key], []).append(r[metric])
    xs = sorted(groups)
    return np.array(xs), np.array([np.mean(groups[x]) for x in xs])


def _plot_metric_grid(records, key, path, xlabel, mark_x=None):
    """Top row: key-cache metrics. Bottom row: value-cache metrics, then both caches."""
    methods = [m for m in METHOD_COLORS if any(r["method"] == m for r in records)]
    metrics = [m for m in METRIC_TITLES if m in records[0]]
    ncols = (len(metrics) + 1) // 2
    fig, axes = plt.subplots(2, ncols, figsize=(5.5 * ncols, 7.5), squeeze=False)
    for ax, metric in zip(axes.flat, metrics):
        for method in methods:
            xs, ys = _mean_by(records, key, metric, method)
            # Markers on every point for short x ranges; spaced out for long ones.
            ax.plot(xs, ys, color=METHOD_COLORS[method], linewidth=2,
                    marker=METHOD_MARKERS[method], markersize=4,
                    markevery=1 if len(xs) <= 40 else max(1, len(xs) // 10),
                    label=METHOD_LABELS[method])
        for x in mark_x or []:
            ax.axvline(x, color="#999999", linewidth=1, linestyle="--")
        ax.set_title(METRIC_TITLES[metric])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Relative error")
        _style_axis(ax)
    for ax in axes.flat[len(metrics):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(methods), frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_errors_by_layer(records, path):
    """Each metric averaged over heads and positions, plotted against layer index."""
    _plot_metric_grid(records, "layer", path, "Layer")


def plot_errors_by_position(records, path, residual_length=None):
    """Each metric averaged over layers and heads, plotted against query position.

    Dashed lines mark positions where the key residual has just been quantized
    (prefix length is a multiple of residual_length).
    """
    positions = sorted({r["position"] for r in records})
    mark_x = None
    if residual_length:
        mark_x = [p for p in positions if (p + 1) % residual_length == 0]
    _plot_metric_grid(records, "position", path, "Query position in prompt", mark_x)
