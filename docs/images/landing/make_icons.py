"""Generate the task illustrations of the landing page, icons/*.svg.

Each illustration is a small diagram of what the task does to a time series. They have
no colours of their own and are styled and animated by docs/_static/css/landing.css.
The classes starting with "i-" pick a colour ("i-a", "i-b", "i-ink") or an animation
played when the card is hovered ("i-draw", "i-pop", "i-rise", ...), delayed by the
"--d" custom property.

Run from this directory: python make_icons.py
"""

from pathlib import Path

import numpy as np

from aeon.distances import dtw_alignment_path

WIDTH, HEIGHT = 160, 90
T = np.linspace(0, 1, 80)


def _fmt(v):
    return f"{v:.1f}".rstrip("0").rstrip(".")


def line(xs, ys, cls, delay=None, draw=False):
    """Return a polyline through the points, optionally drawn on hover."""
    points = " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in zip(xs, ys))
    attrs = f'class="i-line {cls}{" i-draw" if draw else ""}"'
    if draw:
        attrs += ' pathLength="1"'
    if delay is not None:
        attrs += f' style="--d:{delay}s"'
    return f'<polyline {attrs} points="{points}"/>'


def shape(tag, cls, delay=None, **attrs):
    """Return any other SVG element, attribute names use "_" instead of "-"."""
    parts = [f'class="{cls}"']
    if delay is not None:
        parts.append(f'style="--d:{delay}s"')
    parts += [
        f'{k.replace("_", "-")}="{_fmt(v) if not isinstance(v, str) else v}"'
        for k, v in attrs.items()
    ]
    return f"<{tag} {' '.join(parts)}/>"


def span(x0, x1, t=T):
    """Map t in [0, 1] to the horizontal range [x0, x1]."""
    return x0 + (x1 - x0) * t


def bump(t, centre, width):
    """Return a Gaussian bump of height one."""
    return np.exp(-(((t - centre) / width) ** 2))


def classification():
    """Draw two series of different shapes, each sent to its class label."""
    a = 30 - 13 * np.sin(2 * np.pi * T)
    b = 62 - 9 * np.tanh(4 * np.sin(2 * np.pi * 2.5 * T))
    return [
        line(span(10, 92), a, "i-a", 0, draw=True),
        line(span(10, 92), b, "i-b", 0.15, draw=True),
        '<path class="i-line i-ink" d="M100 30h12M100 62h12"/>',
        shape("rect", "i-chip-a i-pop", 0.7, x=120, y=20, width=30, height=20, rx=10),
        shape("rect", "i-chip-b i-pop", 0.85, x=120, y=52, width=30, height=20, rx=10),
        '<text class="i-chip-text i-pop" style="--d:.7s" x="135" y="34.5">A</text>',
        '<text class="i-chip-text i-pop" style="--d:.85s" x="135" y="66.5">B</text>',
    ]


def regression():
    """Draw one series mapped to a value on a continuous scale."""
    y = 48 - 16 * np.sin(2 * np.pi * 1.5 * T) - 10 * T
    ticks = "".join(f"M134 {v}h8" for v in (16, 30.5, 45, 59.5, 74))
    return [
        line(span(10, 96), y, "i-a", 0, draw=True),
        '<path class="i-line i-ink" d="M104 45h12"/>',
        f'<path class="i-line i-ink" d="M134 16v58{ticks}"/>',
        shape("circle", "i-fill-b i-settle", 0.6, cx=134, cy=34, r=6.5),
    ]


def clustering():
    """Draw six unlabelled series gathered into two groups by shape."""
    out = [
        shape("circle", "i-line i-a i-dash i-pop", 0.5, cx=44, cy=45, r=35),
        shape("circle", "i-line i-b i-dash i-pop", 0.65, cx=116, cy=45, r=35),
    ]
    t = T[:40] * 2
    for k, cy in enumerate((28, 45, 62)):
        wave = cy - 7 * np.sin(2 * np.pi * t + 0.5 * k)
        step = cy - 6 * np.tanh(5 * np.sin(2 * np.pi * 1.5 * t + 0.7 * k))
        out.append(line(span(22, 66, t), wave, "i-a i-fade", 0.1 * k))
        out.append(line(span(94, 138, t), step, "i-b i-fade", 0.1 * k + 0.05))
    return out


def anomaly_detection():
    """Draw a periodic series with one spike, flagged."""
    y = 54 - 9 * np.sin(2 * np.pi * 6 * T) - 34 * bump(T, 0.46, 0.018)
    xs = span(10, 150)
    odd = np.flatnonzero(np.abs(T - 0.46) < 0.05)
    peak = int(np.argmin(y))
    return [
        line(xs, y, "i-ink"),
        line(xs[odd], y[odd], "i-b", 0.1, draw=True),
        shape("circle", "i-line i-b i-pulse", 0.5, cx=xs[peak], cy=y[peak], r=8),
    ]


def forecasting():
    """Draw a series continued past the present, with growing uncertainty."""

    def f(t):
        return 58 - 24 * t - 9 * np.sin(2 * np.pi * 2.6 * t)

    past, future = T * 0.62, 0.62 + T * 0.38
    xs_f, ys_f = span(10, 150, future), f(future)
    spread = 16 * (future - 0.62) / 0.38
    cone = " ".join(
        f"{_fmt(x)},{_fmt(v)}"
        for x, v in list(zip(xs_f, ys_f - spread))
        + list(zip(xs_f, ys_f + spread))[::-1]
    )
    return [
        f'<polygon class="i-soft-b i-fade" style="--d:.5s" points="{cone}"/>',
        '<path class="i-line i-ink i-dash" d="M96.8 12v66"/>',
        line(span(10, 150, past), f(past), "i-a", 0, draw=True),
        line(xs_f, ys_f, "i-b i-dash i-reveal", 0.5),
    ]


def segmentation():
    """Draw a series with three regimes and the two change points."""
    xs = span(10, 150)
    y = np.where(
        T < 0.34,
        48 - 8 * np.sin(2 * np.pi * 3 * T / 0.34),
        np.where(
            T < 0.7,
            48 - 18 * np.sin(2 * np.pi * 9 * (T - 0.34) / 0.36),
            30 - 4 * np.sin(2 * np.pi * 2 * (T - 0.7) / 0.3),
        ),
    )
    cuts = [10 + 140 * 0.34, 10 + 140 * 0.7]
    return [
        shape("rect", "i-soft-a", x=10, y=12, width=cuts[0] - 10, height=66, rx=4),
        shape("rect", "i-soft-c", x=cuts[0], y=12, width=cuts[1] - cuts[0], height=66),
        shape(
            "rect", "i-soft-b", x=cuts[1], y=12, width=150 - cuts[1], height=66, rx=4
        ),
        line(xs, y, "i-ink-strong"),
        *[
            f'<path class="i-line i-b i-dash i-drop" style="--d:{delay}s" '
            f'd="M{_fmt(cut)} 8v74"/>'
            for cut, delay in zip(cuts, (0.1, 0.35))
        ],
    ]


def similarity_search():
    """Draw a query window and its best match further along the series."""
    y = 58 - 4 * np.sin(2 * np.pi * 9 * T) - 30 * bump(T, 0.24, 0.035)
    y = y - 30 * bump(T, 0.7, 0.035)
    return [
        line(span(10, 150), y, "i-ink-strong"),
        shape("rect", "i-line i-a", x=28.6, y=16, width=30, height=56, rx=5),
        shape("rect", "i-line i-b i-seek", 0.1, x=93, y=16, width=30, height=56, rx=5),
    ]


def transformations():
    """Draw a series turned into a vector of features."""
    y = 46 - 18 * np.sin(2 * np.pi * 1.5 * T) * (1 - 0.5 * T)
    out = [
        line(span(10, 74), y, "i-a", 0, draw=True),
        '<path class="i-line i-ink" d="M82 45h14m-5-5l5 5l-5 5"/>',
    ]
    for k, (h, cls) in enumerate(
        [(42, "i-fill-b"), (22, "i-fill-a"), (54, "i-fill-b"), (32, "i-fill-a")]
    ):
        out.append(
            shape(
                "rect",
                f"{cls} i-rise",
                0.5 + 0.1 * k,
                x=106 + 12 * k,
                y=76 - h,
                width=8,
                height=h,
                rx=2,
            )
        )
    return out


def distances():
    """Draw two series linked by their DTW alignment path, computed with aeon."""
    n = 28
    t = np.linspace(0, 1, n)
    top = bump(t, 0.34, 0.13)
    bottom = bump(t, 0.6, 0.17)
    xs = span(12, 148, t)
    y_top, y_bottom = 34 - 22 * top, 80 - 22 * bottom
    path, _ = dtw_alignment_path(top, bottom)
    links = "".join(
        f"M{_fmt(xs[i])} {_fmt(y_top[i])}L{_fmt(xs[j])} {_fmt(y_bottom[j])}"
        for i, j in path[::2]
    )
    return [
        f'<path class="i-line i-ink i-thin i-fade" style="--d:.5s" d="{links}"/>',
        line(xs, y_top, "i-a", 0, draw=True),
        line(xs, y_bottom, "i-b", 0.15, draw=True),
    ]


def networks():
    """Draw a small fully connected network."""
    layers = [
        (22, [25, 45, 65]),
        (62, [15, 35, 55, 75]),
        (102, [15, 35, 55, 75]),
        (140, [35, 55]),
    ]
    edges = "".join(
        f"M{x0} {y0}L{x1} {y1}"
        for (x0, ys0), (x1, ys1) in zip(layers, layers[1:])
        for y0 in ys0
        for y1 in ys1
    )
    out = [f'<path class="i-line i-ink i-thin" d="{edges}"/>']
    for k, (x, ys) in enumerate(layers):
        cls = ["i-fill-a", "i-fill-ink", "i-fill-ink", "i-fill-b"][k]
        out += [
            shape("circle", f"{cls} i-pop", 0.15 * k, cx=x, cy=v, r=5.5) for v in ys
        ]
    return out


def datasets():
    """Draw a collection of series, stacked."""
    t = T[:50] / T[49]
    out = []
    for k, cls in enumerate(["i-ink", "i-b", "i-a"]):
        dx, dy = 26 - 13 * k, 10 + 9 * k
        wave = dy + 27 - 9 * np.sin(2 * np.pi * (1.5 + 0.5 * k) * t + k)
        out.append(
            f'<g class="i-fan" style="--d:{0.08 * k}s;--fan:{_fmt((1 - k) * 9)}px">'
            + shape("rect", "i-card", x=30 + dx, y=dy, width=92, height=54, rx=7)
            + line(span(40 + dx, 112 + dx, t), wave, cls)
            + "</g>"
        )
    return out


def benchmarking():
    """Draw a critical difference diagram ranking four estimators."""
    ranks = [
        (38, 46, 14, "i-fill-a"),
        (66, 62, 14, "i-fill-ink"),
        (84, 62, 146, "i-fill-ink"),
        (124, 46, 146, "i-fill-b"),
    ]
    ticks = "".join(f"M{x} 20v8" for x in (20, 50, 80, 110, 140))
    out = [
        f'<path class="i-line i-ink" d="M20 24h120{ticks}"/>',
        '<path class="i-line i-ink-strong i-thick i-fade" style="--d:.7s" '
        'd="M62 33h26"/>',
    ]
    for k, (x, y, x_end, cls) in enumerate(ranks):
        out.append(
            f'<g class="i-fade" style="--d:{0.15 * k}s">'
            f'<path class="i-line i-ink i-thin" d="M{x} 24V{y}H{x_end}"/>'
            + shape("circle", cls, cx=x, cy=24, r=4.5)
            + "</g>"
        )
    return out


ICONS = [
    classification,
    regression,
    clustering,
    anomaly_detection,
    forecasting,
    segmentation,
    similarity_search,
    transformations,
    distances,
    networks,
    datasets,
    benchmarking,
]

out_dir = Path(__file__).parent / "icons"
out_dir.mkdir(exist_ok=True)
for icon in ICONS:
    body = "\n".join(icon())
    svg = (
        f'<svg class="aeon-icon" viewBox="0 0 {WIDTH} {HEIGHT}" aria-hidden="true" '
        f'focusable="false" xmlns="http://www.w3.org/2000/svg">\n{body}\n</svg>\n'
    )
    (out_dir / f"{icon.__name__}.svg").write_text(svg, encoding="utf-8")
