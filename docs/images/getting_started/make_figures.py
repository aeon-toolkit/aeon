"""Generate the figures of the getting started page, *.svg in this directory.

Each data figure shows the output of the code example it follows in
docs/getting_started.md, so the examples are run here as they are written there. The
other figures are diagrams of the data structures.

The figures have no colours of their own. They are styled by
docs/_static/css/getting_started.css, where the classes starting with "f-" pick a
colour ("f-a", "f-b", "f-ink"), a fill ("f-fill-a", "f-soft-a") or a line style
("f-thin", "f-bold", "f-faint", "f-dash"). This way they follow the light and dark
themes. Text is limited to tick labels and short names, the captions are written in
the page.

Run from this directory: python make_figures.py
A few figures only:      python make_figures.py forecasting acf
The anomaly detection figure needs the stumpy package.
"""

import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
WIDTH = 600
# left and right edges of the panels with tick labels on their left
LEFT, RIGHT = 56, 584


def _fmt(v):
    return f"{v:.1f}".rstrip("0").rstrip(".")


def _points(xs, ys):
    return " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in zip(xs, ys))


def text(x, y, s, cls="f-text", anchor="middle", rotate=None):
    """Return a text element, optionally rotated around its anchor."""
    attrs = f'class="{cls}" x="{_fmt(x)}" y="{_fmt(y)}" text-anchor="{anchor}"'
    if rotate is not None:
        attrs += f' transform="rotate({rotate} {_fmt(x)} {_fmt(y)})"'
    return f"<text {attrs}>{s}</text>"


def rect(x, y, w, h, cls, rx=4, opacity=None):
    """Return a rounded rectangle."""
    style = "" if opacity is None else f' style="fill-opacity:{opacity:.2f}"'
    return (
        f'<rect class="{cls}" x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(w)}" '
        f'height="{_fmt(h)}" rx="{rx}"{style}/>'
    )


def arrow(x0, x1, y):
    """Return a horizontal arrow pointing right."""
    d = f"M{_fmt(x0)} {_fmt(y)}H{_fmt(x1)}m-6 -5l6 5l-6 5"
    return f'<path class="f-line f-ink-strong" d="{d}"/>'


def cells(x0, y0, n_rows, n_cols, size, cls, gap=2, opacity=None):
    """Return a grid of squares, one per value of an array.

    ``cls`` is one class or one class per row, ``opacity`` an optional array of fill
    opacities of shape (n_rows, n_cols).
    """
    row_cls = [cls] * n_rows if isinstance(cls, str) else cls
    return "".join(
        rect(
            x0 + j * (size + gap),
            y0 + i * (size + gap),
            size,
            size,
            row_cls[i],
            rx=min(3, size / 4),
            opacity=None if opacity is None else opacity[i][j],
        )
        for i in range(n_rows)
        for j in range(n_cols)
    )


def box(x, y, w, h, label):
    """Return a rounded box with a centred label, a step of a diagram."""
    return rect(x, y, w, h, "f-box", rx=10) + text(
        x + w / 2, y + h / 2 + 4.5, label, cls="f-code"
    )


class Panel:
    """Map data coordinates to a rectangle of the figure."""

    def __init__(self, x0, y0, w, h, xlim, ylim):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.xlim, self.ylim = xlim, ylim

    def x(self, v):
        """Return the horizontal position of the values."""
        lo, hi = self.xlim
        return self.x0 + (np.asarray(v, dtype=float) - lo) / (hi - lo) * self.w

    def y(self, v):
        """Return the vertical position of the values."""
        lo, hi = self.ylim
        ratio = (np.asarray(v, dtype=float) - lo) / (hi - lo)
        return self.y0 + self.h - ratio * self.h

    def line(self, xs, ys, cls):
        """Return a line through the points."""
        points = _points(self.x(xs), self.y(ys))
        return f'<polyline class="f-line {cls}" points="{points}"/>'

    def dots(self, xs, ys, cls, r=3.5):
        """Return one dot per point."""
        return "".join(
            f'<circle class="{cls}" cx="{_fmt(x)}" cy="{_fmt(y)}" r="{r}"/>'
            for x, y in zip(self.x(xs), self.y(ys))
        )

    def vline(self, x, cls):
        """Return a vertical line over the height of the panel."""
        d = f"M{_fmt(float(self.x(x)))} {_fmt(self.y0)}v{_fmt(self.h)}"
        return f'<path class="f-line {cls}" d="{d}"/>'

    def band(self, x_from, x_to, cls):
        """Return a rectangle over the height of the panel."""
        a, b = float(self.x(x_from)), float(self.x(x_to))
        return rect(a, self.y0, b - a, self.h, cls)

    def baseline(self):
        """Return the horizontal axis."""
        d = f"M{_fmt(self.x0)} {_fmt(self.y0 + self.h)}h{_fmt(self.w)}"
        return f'<path class="f-line f-ink f-thin" d="{d}"/>'

    def xticks(self, values, labels):
        """Return tick marks and labels under the panel."""
        y = self.y0 + self.h
        parts = []
        for px, label in zip(self.x(values), labels):
            parts.append(
                f'<path class="f-line f-ink f-thin" d="M{_fmt(px)} {_fmt(y)}v5"/>'
            )
            # a label at the right edge ends there, the text is larger on phones
            if px > WIDTH - 30:
                parts.append(text(WIDTH - 6, y + 21, label, anchor="end"))
            else:
                parts.append(text(px, y + 21, label))
        return "".join(parts)

    def yticks(self, values, labels):
        """Return labels on the left of the panel and a faint line for each."""
        x0, w = _fmt(self.x0), _fmt(self.w)
        return "".join(
            f'<path class="f-line f-grid" d="M{x0} {_fmt(py)}h{w}"/>'
            + text(self.x0 - 8, py + 4.5, label, anchor="end")
            for py, label in zip(self.y(values), labels)
        )


def save(name, height, title, parts):
    """Write one figure."""
    svg = (
        f'<svg viewBox="0 0 {WIDTH} {height}" role="img" '
        f'aria-labelledby="fig-{name}" xmlns="http://www.w3.org/2000/svg">'
        f'<title id="fig-{name}">{title}</title>{"".join(parts)}</svg>\n'
    )
    (OUT / f"{name}.svg").write_text(svg, encoding="utf-8")


# The airline series is monthly and starts in January 1949.
def _year(year):
    return (year - 1949) * 12


# -- single series examples --


def anomaly_detection():
    """Draw the airline series above its STOMP anomaly score."""
    from aeon.anomaly_detection.series.distance_based import STOMP
    from aeon.datasets import load_airline

    y = load_airline()
    stomp = STOMP(window_size=12)
    scores = stomp.fit_predict(y)

    t = np.arange(len(y))
    top = Panel(LEFT, 34, RIGHT - LEFT, 104, (0, 143), (80, 650))
    bottom = Panel(LEFT, 176, RIGHT - LEFT, 74, (0, 143), (0, scores.max() * 1.05))
    # the 12 months with the highest average score
    start = int(np.argmax(np.convolve(scores, np.ones(12), mode="valid")))
    years = [1949, 1952, 1955, 1958, 1961]
    return (
        292,
        "The airline series and its anomaly score, highest for one of the years",
        [
            top.band(start, start + 11, "f-soft-b"),
            bottom.band(start, start + 11, "f-soft-b"),
            top.baseline(),
            bottom.baseline(),
            top.line(t, y, "f-a"),
            bottom.line(t, scores, "f-b"),
            bottom.xticks([_year(v) for v in years[:-1]] + [143], map(str, years)),
            text(LEFT, 24, "passengers", anchor="start"),
            text(LEFT, 166, "anomaly score", anchor="start"),
        ],
    )


def forecasting():
    """Draw the last five years of the airline series and six forecast months."""
    from aeon.datasets import load_airline
    from aeon.forecasting.stats import ETS

    y = load_airline()
    ets = ETS(
        trend_type="additive", seasonality_type="multiplicative", seasonal_period=12
    )
    pred = ets.iterative_forecast(y, prediction_horizon=6)

    first = _year(1956)
    t = np.arange(first, len(y))
    t_pred = np.arange(len(y), len(y) + 6)
    panel = Panel(LEFT, 20, RIGHT - LEFT, 165, (first, len(y) + 6), (250, 650))
    years = [1956, 1957, 1958, 1959, 1960, 1961]
    return (
        228,
        "Monthly airline passengers from 1956 to 1960 and the six forecast months",
        [
            panel.band(len(y) - 0.5, len(y) + 6, "f-soft-b"),
            panel.yticks([300, 450, 600], ["300", "450", "600"]),
            panel.baseline(),
            panel.xticks([_year(v) for v in years], map(str, years)),
            panel.line(t, y[first:], "f-a"),
            panel.line(np.r_[t[-1], t_pred], np.r_[y[-1], pred], "f-b"),
            panel.dots(t_pred, pred, "f-fill-b"),
        ],
    )


def segmentation():
    """Draw the airline series split at the change point found by ClaSP."""
    from aeon.datasets import load_airline
    from aeon.segmentation import ClaSPSegmenter

    y = load_airline()
    clasp = ClaSPSegmenter()
    change_point = int(clasp.fit_predict(y)[0])

    t = np.arange(len(y))
    panel = Panel(LEFT, 20, RIGHT - LEFT, 165, (0, 143), (80, 650))
    years = [1949, 1952, 1955, 1958, 1961]
    return (
        228,
        f"The airline series split in two regions at index {change_point}",
        [
            panel.band(0, change_point, "f-soft-a"),
            panel.band(change_point, 143, "f-soft-b"),
            panel.yticks([200, 400, 600], ["200", "400", "600"]),
            panel.baseline(),
            panel.xticks([_year(v) for v in years[:-1]] + [143], map(str, years)),
            panel.line(t, y, "f-ink-strong"),
            panel.vline(change_point, "f-ink-strong f-dash"),
            text(float(panel.x(change_point)) + 8, 36, "change point", anchor="start"),
        ],
    )


def distances():
    """Draw two series of unequal length and their DTW alignment."""
    from aeon.datasets import load_japanese_vowels
    from aeon.distances import dtw_alignment_path

    X, _ = load_japanese_vowels()
    path, _ = dtw_alignment_path(X[0], X[1])

    a, b = X[0][0], X[1][0]
    ylim = (min(a.min(), b.min()), max(a.max(), b.max()))
    xlim = (0, max(len(a), len(b)) - 1)
    top = Panel(40, 22, 520, 80, xlim, ylim)
    bottom = Panel(40, 150, 520, 80, xlim, ylim)
    ta, tb = np.arange(len(a)), np.arange(len(b))
    links = "".join(
        f'<path class="f-line f-ink f-thin" d="M{_fmt(float(top.x(i)))} '
        f"{_fmt(float(top.y(a[i])))}L{_fmt(float(bottom.x(j)))} "
        f'{_fmt(float(bottom.y(b[j])))}"/>'
        for i, j in path
    )
    return (
        252,
        "Two series of 20 and 26 time points and the points matched by DTW",
        [
            links,
            top.line(ta, a, "f-a"),
            top.dots(ta, a, "f-fill-a", r=3),
            bottom.line(tb, b, "f-b"),
            bottom.dots(tb, b, "f-fill-b", r=3),
        ],
    )


# -- collection examples --


def classification():
    """Draw days of electricity demand coloured by class."""
    from aeon.datasets import load_italy_power_demand

    X_train, y_train = load_italy_power_demand(split="train")

    hours = np.arange(24)
    panel = Panel(LEFT, 20, RIGHT - LEFT, 165, (0, 23), (X_train.min(), X_train.max()))
    parts = [
        panel.baseline(),
        panel.xticks([0, 6, 12, 18, 23], ["0h", "6h", "12h", "18h", "23h"]),
        text(LEFT - 14, 102, "demand", rotate=-90),
    ]
    for label, cls in (("1", "f-a"), ("2", "f-b")):
        for series in X_train[y_train == label][:12, 0]:
            parts.append(panel.line(hours, series, f"{cls} f-thin f-faint"))
    for label, cls in (("1", "f-a"), ("2", "f-b")):
        mean = X_train[y_train == label][:, 0].mean(axis=0)
        parts.append(panel.line(hours, mean, f"{cls} f-bold"))
    return (
        228,
        "Days of electricity demand, the two classes have different shapes",
        parts,
    )


def regression():
    """Draw the first three test series with their true and predicted values."""
    from aeon.datasets import load_covid_3month
    from aeon.regression.convolution_based import RocketRegressor

    X_train, y_train = load_covid_3month(split="train")
    X_test, y_test = load_covid_3month(split="test")
    reg = RocketRegressor(random_state=0)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    t = np.arange(X_test.shape[2])
    parts = []
    for k in range(3):
        series = X_test[k, 0]
        panel = Panel(24 + k * 192, 22, 168, 104, (0, t[-1]), (0, series.max()))
        parts += [
            rect(panel.x0 - 8, 10, 184, 126, "f-card", rx=8),
            panel.line(t, series, "f-a"),
            text(
                panel.x0 + 84, 160, f"predicted {y_pred[k]:.3f}", cls="f-text f-strong"
            ),
            text(panel.x0 + 84, 182, f"true {y_test[k]:.3f}"),
        ]
    return (
        198,
        "Three test series, each with the value predicted by the regressor and the "
        "true value",
        parts,
    )


def clustering():
    """Draw the members and the centre of each cluster."""
    from aeon.clustering import TimeSeriesKMeans
    from aeon.datasets import load_arrow_head

    X, y = load_arrow_head()
    kmeans = TimeSeriesKMeans(
        n_clusters=3, distance="dtw", averaging_method="mean", random_state=0
    )
    kmeans.fit(X)

    t = np.arange(X.shape[2])
    parts = []
    for k, cls in enumerate(("f-a", "f-b", "f-ink-strong")):
        panel = Panel(24 + k * 192, 18, 168, 132, (0, t[-1]), (X.min(), X.max()))
        members = X[kmeans.labels_ == k][:, 0]
        parts.append(rect(panel.x0 - 8, 10, 184, 148, "f-card", rx=8))
        # every third point is enough for the thin lines and keeps the file small
        parts += [
            panel.line(t[::3], s[::3], f"{cls} f-thin f-faint") for s in members[:12]
        ]
        parts.append(panel.line(t, kmeans.cluster_centers_[k, 0], f"{cls} f-bold"))
        parts.append(text(panel.x0 + 84, 182, f"{len(members)} series"))
    return (
        198,
        "The series of the three clusters, with the centre of each cluster in bold",
        parts,
    )


def similarity_search():
    """Draw the query over its best match in the series."""
    from aeon.similarity_search.subsequence import MASS

    X = np.array([[[1, 1, 2, 4, 6, 6, 7, 5, 3, 2]]])
    q = np.array([[0, 1, 2, 2]])
    snn = MASS(length=4).fit(X)
    indexes, _ = snn.predict(q, k=1)

    start = int(indexes[0][1])
    match = np.arange(start, start + 4)
    t = np.arange(X.shape[2])
    panel = Panel(LEFT, 20, RIGHT - LEFT, 160, (-0.5, 9.5), (-0.6, 7.6))
    return (
        224,
        "A series of ten points, the query and the closest subsequence of the series",
        [
            panel.band(start - 0.35, start + 3.35, "f-soft-a"),
            panel.yticks([0, 2, 4, 6], ["0", "2", "4", "6"]),
            panel.baseline(),
            panel.xticks(t, map(str, t)),
            panel.line(t, X[0, 0], "f-ink-strong"),
            panel.dots(t, X[0, 0], "f-fill-ink"),
            panel.line(match, X[0, 0, match], "f-a f-bold"),
            panel.dots(match, X[0, 0, match], "f-fill-a"),
            panel.line(match, q[0], "f-b f-bold"),
            panel.dots(match, q[0], "f-fill-b"),
        ],
    )


# -- transformations and pipelines --


def acf():
    """Draw the autocorrelation of the airline series as bars."""
    from aeon.datasets import load_airline
    from aeon.transformations.series import AutoCorrelationSeriesTransformer

    acf = AutoCorrelationSeriesTransformer()
    y = load_airline()
    res = acf.fit_transform(y)

    values = res[0]
    lags = np.arange(1, len(values) + 1)
    panel = Panel(LEFT, 20, RIGHT - LEFT, 160, (0.2, len(values) + 0.8), (0, 1))
    width = panel.w / (len(values) + 0.6) * 0.72
    bars = "".join(
        rect(
            float(panel.x(lag)) - width / 2,
            float(panel.y(v)),
            width,
            float(panel.y(0) - panel.y(v)),
            "f-fill-b" if lag % 12 == 0 else "f-fill-a",
            rx=2,
        )
        for lag, v in zip(lags, values)
    )
    return (
        224,
        "Autocorrelation of the airline series, with a peak every 12 months",
        [
            panel.yticks([0.5, 1], ["0.5", "1"]),
            bars,
            panel.baseline(),
            panel.xticks([1, 12, 24, 36], ["lag 1", "12", "24", "36"]),
        ],
    )


def catch22():
    """Draw four series turned into four rows of 22 features."""
    from aeon.transformations.collection.feature_based import Catch22

    X = np.random.RandomState(0).random(size=(4, 1, 10))
    c22 = Catch22(replace_nans=True)
    features = c22.fit_transform(X)

    lo, hi = features.min(axis=0), features.max(axis=0)
    scaled = (features - lo) / np.where(hi > lo, hi - lo, 1)
    parts = []
    for i in range(4):
        panel = Panel(28, 26 + i * 32, 120, 22, (0, 9), (0, 1))
        parts.append(panel.line(np.arange(10), X[i, 0], "f-a"))
    grid_x, size = 312, 10
    parts += [
        arrow(160, 186, 86),
        box(194, 64, 84, 44, "Catch22"),
        arrow(286, 304, 86),
        cells(grid_x, 63, 4, 22, size, "f-fill-a", opacity=0.15 + 0.85 * scaled),
        text(88, 172, "4 series"),
        text(grid_x + 131, 172, "4 rows of 22 features"),
    ]
    return (
        190,
        "Catch22 turns four series into four rows of 22 features",
        parts,
    )


def padder():
    """Draw two series of unequal length, the shorter one padded with zeros."""
    from aeon.testing.data_generation import make_example_3d_numpy_list
    from aeon.transformations.collection.unequal_length import Padder

    X, _ = make_example_3d_numpy_list(
        n_cases=2, min_n_timepoints=8, max_n_timepoints=12, random_state=0
    )
    pad = Padder(padded_length=12, fill_value=0)
    padded = pad.fit_transform(X)

    parts = []
    for i, y0 in enumerate((22, 118)):
        panel = Panel(LEFT, y0, RIGHT - LEFT, 62, (-0.5, 11.5), (-0.4, 4.2))
        n = X[i].shape[1]
        t = np.arange(12)
        parts += [
            panel.baseline(),
            panel.line(t[:n], padded[i, 0, :n], "f-a"),
            panel.dots(t[:n], padded[i, 0, :n], "f-fill-a"),
        ]
        if n < 12:
            parts += [
                panel.band(n - 0.45, 11.45, "f-soft-b"),
                panel.line(t[n - 1 :], padded[i, 0, n - 1 :], "f-b f-dash"),
                panel.dots(t[n:], padded[i, 0, n:], "f-fill-b"),
            ]
        if i == 1:
            parts.append(panel.xticks(t, map(str, t)))
    return (
        214,
        "Two series of 12 and 9 time points, the second one padded with three zeros",
        parts,
    )


def pipeline():
    """Draw series going through a transformer and a classifier."""
    t = np.linspace(0, 2 * np.pi, 30)
    parts = [rect(20, 34, 96, 84, "f-card", rx=8)]
    for i, (cls, freq) in enumerate((("f-a", 1), ("f-b", 2), ("f-a", 1.5))):
        panel = Panel(30, 44 + i * 24, 76, 16, (0, t[-1]), (-1, 1))
        parts.append(panel.line(t, np.sin(freq * t), cls))
    parts += [
        arrow(124, 148, 76),
        box(156, 54, 116, 44, "Catch22"),
        arrow(280, 304, 76),
        box(312, 54, 156, 44, "RandomForest"),
        arrow(476, 500, 76),
        rect(510, 46, 60, 26, "f-fill-a", rx=13),
        rect(510, 80, 60, 26, "f-fill-b", rx=13),
        text(540, 64, "1", cls="f-chip-text"),
        text(540, 98, "2", cls="f-chip-text"),
        text(68, 146, "time series"),
        text(214, 146, "transformer"),
        text(390, 146, "classifier"),
        text(540, 146, "labels"),
    ]
    return (
        166,
        "A pipeline: time series, the Catch22 transformer, a classifier, the labels",
        parts,
    )


# -- data structures --


def _wave(n, phase, freq=1.0):
    t = np.linspace(0, 1, n)
    return 0.5 + 0.38 * np.sin(2 * np.pi * (freq * t + phase)) + 0.1 * np.cos(9 * t)


def series():
    """Draw a univariate and a multivariate series above their arrays."""
    n, size, gap = 12, 17, 3
    step = size + gap
    parts = []
    for x0, channels, title in (
        (40, [("f-a", "f-fill-a", "f-soft-a")], "univariate"),
        (
            336,
            [
                ("f-a", "f-fill-a", "f-soft-a"),
                ("f-b", "f-fill-b", "f-soft-b"),
                ("f-ink-strong", "f-fill-ink", "f-soft-ink"),
            ],
            "multivariate",
        ),
    ):
        # one dot per time point, above the cell that stores it
        centres = x0 + size / 2 + step * np.arange(n)
        parts.append(text(x0 + (n * step - gap) / 2, 24, title))
        for i, (line_cls, dot_cls, _) in enumerate(channels):
            values = _wave(n, 0.23 * i, 1 + 0.35 * i)
            height = 78 / len(channels)
            ys = 38 + i * height + (1 - values) * (height - 8)
            parts.append(
                f'<polyline class="f-line {line_cls}" points="{_points(centres, ys)}"/>'
            )
            parts += [
                f'<circle class="{dot_cls}" cx="{_fmt(x)}" cy="{_fmt(y)}" r="3"/>'
                for x, y in zip(centres, ys)
            ]
        rows_y = 136 if len(channels) == 3 else 136 + step
        parts.append(
            cells(x0, rows_y, len(channels), n, size, [c[2] for c in channels], gap)
        )
        parts.append(text(x0 + (n * step - gap) / 2, 218, "n_timepoints", cls="f-code"))
        if len(channels) > 1:
            parts.append(
                text(
                    x0 - 12,
                    136 + 1.5 * step - gap / 2,
                    "n_channels",
                    "f-code",
                    rotate=-90,
                )
            )
    return (
        232,
        "A univariate series is one row of values, a multivariate series has one row "
        "per channel",
        parts,
    )


def _case(x, y, n_cols, size, gap, cls):
    """Return one case of a collection: a card holding a 2 channel array."""
    step = size + gap
    return rect(
        x - 6, y - 6, n_cols * step - gap + 12, 2 * step - gap + 12, "f-card"
    ) + (cells(x, y, 2, n_cols, size, cls, gap))


def collection():
    """Draw an equal length collection as a 3D array and an unequal one as a list."""
    size, gap = 14, 3
    step = size + gap
    parts = [text(150, 24, "equal length"), text(450, 24, "unequal length")]
    # three cases stacked from back to front
    for k in (2, 1, 0):
        cls = ["f-soft-a", "f-soft-b"] if k == 0 else ["f-soft-ink", "f-soft-ink"]
        parts.append(_case(64 + k * 16, 118 - k * 22, 10, size, gap, cls))
    parts += [
        text(64 + 5 * step, 182, "n_timepoints", cls="f-code"),
        text(46, 118 + step - gap / 2, "n_channels", cls="f-code", rotate=-90),
        text(288, 98, "n_cases", cls="f-code", rotate=-54),
    ]
    for k, n_cols in enumerate((10, 7, 12)):
        parts.append(
            _case(352, 52 + k * 56, n_cols, size, gap, ["f-soft-a", "f-soft-b"])
        )
    return (
        226,
        "An equal length collection is one 3D array, an unequal length collection is "
        "a list of 2D arrays",
        parts,
    )


FIGURES = {
    f.__name__: f
    for f in (
        series,
        anomaly_detection,
        forecasting,
        segmentation,
        distances,
        collection,
        classification,
        regression,
        clustering,
        similarity_search,
        acf,
        catch22,
        padder,
        pipeline,
    )
}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGURES:
        save(name, *FIGURES[name]())
