"""Generate the landing page hero panel, hero.html.

The panel shows six days of the solar power series shipped with aeon, seen through
four tasks. Every overlay is computed with aeon: an ETS forecast of the last day, the
STOMP anomaly scores, the MASS nearest neighbours of one day and a nearest neighbour
classification of the last two days. The output has no colours of its own, it is
styled by docs/_static/css/landing.css so that it follows the light and dark themes.
The tabs are driven by docs/_static/js/landing.js, both files are added to the page by
conf.py.

STOMP requires the stumpy soft dependency. Run from this directory:
python make_hero.py
"""

from pathlib import Path

import numpy as np

from aeon.anomaly_detection.series.distance_based import STOMP
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.datasets import load_solar
from aeon.forecasting.stats import ETS
from aeon.similarity_search.subsequence import MASS

DAY = 48  # the series is half-hourly
WIDTH, HEIGHT = 720, 300
PAD_X, PLOT_TOP, PLOT_BOTTOM = 14, 18, 206
STRIP_TOP, STRIP_BOTTOM = 226, 252
LABEL_Y = 284

y = np.asarray(load_solar(), dtype=float)
n_days = len(y) // DAY
y = y[: n_days * DAY]
n = len(y)
y_max = y.max() * 1.05


def _x(t):
    return PAD_X + (WIDTH - 2 * PAD_X) * t / (n - 1)


def _y(v):
    return PLOT_BOTTOM - (PLOT_BOTTOM - PLOT_TOP) * v / y_max


def _points(values, start=0):
    return " ".join(f"{_x(start + i):.1f},{_y(v):.1f}" for i, v in enumerate(values))


def _line(values, start, cls, draw=False):
    length = ' pathLength="1"' if draw else ""
    return f'<polyline class="{cls}"{length} points="{_points(values, start)}"/>'


def _text(x, y_pos, label, cls="aeon-hero-label", anchor="middle"):
    return (
        f'<text class="{cls}" x="{x:.1f}" y="{y_pos}" text-anchor="{anchor}">'
        f"{label}</text>"
    )


# -- the four tasks, computed with aeon ------------------------------------------------

# forecasting: learn from all but the last day and predict it
train = y[:-DAY]
forecaster = ETS(seasonality_type="additive", seasonal_period=DAY)
forecaster.fit(train)
forecast = forecaster.iterative_forecast(train, prediction_horizon=DAY)

# anomaly detection: one score per time point
scores = np.asarray(STOMP(window_size=DAY).fit_predict(y), dtype=float)
scores = (scores - scores.min()) / (scores.max() - scores.min())
peak = int(np.argmax(scores))
flagged = np.flatnonzero(scores >= 0.75)

# similarity search: the daylight hours of the fifth day are the query
QUERY_START, QUERY_LENGTH = 4 * DAY + 8, 32
query = y[QUERY_START : QUERY_START + QUERY_LENGTH]
searcher = MASS(length=QUERY_LENGTH, normalize=False)
searcher.fit(y[np.newaxis, np.newaxis, :])
matches, _ = searcher.predict(
    query[np.newaxis, :], k=2, X_index=(0, QUERY_START), exclusion_factor=1.0
)
match_starts = [int(m[1]) for m in matches]

# classification: each day is one series, learn from the first four and label the rest
N_TRAIN = 4
days = y.reshape(n_days, 1, DAY)
totals = days.sum(axis=(1, 2))
# the known labels: a day that produces much less than the typical day is cloudy
day_classes = np.where(totals < 0.6 * np.median(totals), "cloudy", "sunny")
classifier = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="dtw")
classifier.fit(days[:N_TRAIN], day_classes[:N_TRAIN])
day_classes = np.concatenate(
    [day_classes[:N_TRAIN], classifier.predict(days[N_TRAIN:])]
)

# -- the plot --------------------------------------------------------------------------

svg = [
    f'<svg class="aeon-hero-plot" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
    'xmlns="http://www.w3.org/2000/svg" aria-labelledby="aeon-hero-plot-title">',
    '<title id="aeon-hero-plot-title">Six days of half-hourly solar power generation, '
    "annotated with the results of four aeon estimators</title>",
]

for d in range(n_days + 1):
    x = _x(min(d * DAY, n - 1))
    svg.append(
        f'<line class="aeon-hero-grid" x1="{x:.1f}" y1="{PLOT_TOP}" x2="{x:.1f}" '
        f'y2="{PLOT_BOTTOM}"/>'
    )
for d in range(n_days):
    svg.append(_text(_x(d * DAY + DAY / 2), LABEL_Y, f"Day {d + 1}", "aeon-hero-tick"))

svg.append(_line(y, 0, "aeon-hero-base", draw=True))

# forecast lens
svg.append('<g class="aeon-lens" data-lens="forecast">')
x0 = _x(n - DAY)
svg.append(
    f'<rect class="aeon-hero-band" x="{x0:.1f}" y="{PLOT_TOP}" '
    f'width="{_x(n - 1) - x0:.1f}" height="{PLOT_BOTTOM - PLOT_TOP}"/>'
)
svg.append(_line(forecast, n - DAY, "aeon-hero-mark aeon-hero-dashed aeon-b"))
svg.append(_text(_x(n - DAY / 2), STRIP_TOP + 18, "forecast", "aeon-hero-label aeon-b"))
svg.append("</g>")

# anomaly lens
svg.append('<g class="aeon-lens" data-lens="anomaly">')
strip = " ".join(
    f"{_x(i):.1f},{STRIP_BOTTOM - (STRIP_BOTTOM - STRIP_TOP) * s:.1f}"
    for i, s in enumerate(scores)
)
svg.append(
    f'<polygon class="aeon-hero-score" points="{_x(0):.1f},{STRIP_BOTTOM} {strip} '
    f'{_x(n - 1):.1f},{STRIP_BOTTOM}"/>'
)
svg.append(_text(_x(n - 1), STRIP_TOP + 2, "anomaly score", "aeon-hero-tick", "end"))
svg.append(_line(y[flagged[0] : flagged[-1] + 1], flagged[0], "aeon-hero-mark aeon-b"))
svg.append(
    f'<circle class="aeon-hero-ring aeon-b" cx="{_x(peak):.1f}" cy="{_y(y[peak]):.1f}" '
    'r="9"/>'
)
svg.append("</g>")

# search lens
svg.append('<g class="aeon-lens" data-lens="search">')
svg.append(_line(query, QUERY_START, "aeon-hero-mark aeon-a"))
svg.append(
    _text(
        _x(QUERY_START + QUERY_LENGTH / 2),
        STRIP_TOP + 18,
        "query",
        "aeon-hero-label aeon-a",
    )
)
for rank, start in enumerate(match_starts):
    svg.append(
        _line(
            y[start : start + QUERY_LENGTH],
            start,
            f"aeon-hero-mark aeon-b aeon-delay-{rank + 1}",
        )
    )
    label = "best match" if rank == 0 else "next best"
    svg.append(
        _text(
            _x(start + QUERY_LENGTH / 2),
            STRIP_TOP + 18,
            label,
            f"aeon-hero-label aeon-b aeon-delay-{rank + 1}",
        )
    )
svg.append("</g>")

# classify lens
svg.append('<g class="aeon-lens" data-lens="classify">')
x0 = _x(N_TRAIN * DAY)
svg.append(
    f'<rect class="aeon-hero-band aeon-neutral" x="{x0:.1f}" y="{PLOT_TOP}" '
    f'width="{_x(n - 1) - x0:.1f}" height="{PLOT_BOTTOM - PLOT_TOP}"/>'
)
for d in range(n_days):
    colour = "aeon-a" if day_classes[d] == "sunny" else "aeon-b"
    # the predicted days are drawn after the labelled ones
    delay = "" if d < N_TRAIN else " aeon-delay-2"
    # one more point than a day so that consecutive days join up
    stop = min((d + 1) * DAY + 1, n)
    svg.append(_line(y[d * DAY : stop], d * DAY, f"aeon-hero-mark {colour}{delay}"))
    svg.append(
        _text(
            _x(d * DAY + DAY / 2),
            STRIP_TOP + 18,
            day_classes[d],
            f"aeon-hero-label {colour}{delay}",
        )
    )
# between the peaks of the two predicted days, where the series is at zero
svg.append(
    _text(
        _x((N_TRAIN + n_days) * DAY / 2),
        PLOT_TOP + 14,
        "predicted",
        "aeon-hero-tick aeon-delay-2",
    )
)
svg.append("</g>")
svg.append("</svg>")

# -- the panel around the plot ---------------------------------------------------------

LENSES = [
    (
        "forecast",
        "fa-solid fa-arrow-trend-up",
        "Forecast",
        "<code>ETS</code> learns from the first five days and predicts the sixth.",
        "examples/forecasting/forecasting.html",
    ),
    (
        "anomaly",
        "fa-solid fa-bolt",
        "Detect anomalies",
        "<code>STOMP</code> scores every time point. The low third day stands out.",
        "examples/anomaly_detection/anomaly_detection.html",
    ),
    (
        "search",
        "fa-solid fa-magnifying-glass",
        "Search",
        "<code>MASS</code> takes day 5 as the query and finds the closest matches.",
        "examples/similarity_search/similarity_search.html",
    ),
    (
        "classify",
        "fa-solid fa-tags",
        "Classify",
        "<code>KNeighborsTimeSeriesClassifier</code> learns from four labelled days "
        "and labels the last two.",
        "examples/classification/classification.html",
    ),
]

html = [
    "<!-- generated by docs/images/landing/make_hero.py, do not edit by hand -->",
    f'<div class="aeon-hero-panel" data-lens="{LENSES[0][0]}">',
    '<div class="aeon-hero-tabs" role="tablist" '
    'aria-label="What aeon can do with this series">',
]
for i, (key, icon, name, _, _) in enumerate(LENSES):
    html.append(
        f'<button class="aeon-hero-tab" type="button" role="tab" data-lens="{key}" '
        f'id="aeon-hero-tab-{key}" aria-controls="aeon-hero-view" '
        f'aria-selected="{"true" if i == 0 else "false"}">'
        f'<i class="{icon}" aria-hidden="true"></i><span>{name}</span></button>'
    )
html.append("</div>")
html.append('<div class="aeon-hero-view" id="aeon-hero-view" role="tabpanel">')
html.extend(svg)
for key, _, _, caption, link in LENSES:
    html.append(
        f'<p class="aeon-hero-caption" data-lens="{key}">{caption} '
        f'<a href="{link}">See the example</a></p>'
    )
html.append("</div>")
html.append(
    '<div class="aeon-hero-source"><p>Half-hourly solar power generation over six '
    "days, from <code>aeon.datasets.load_solar</code>. All four results are computed "
    "with <code>aeon</code>.</p>"
    # shown by landing.js, which is also what moves the panel from tab to tab
    '<button class="aeon-hero-pause" type="button" hidden>'
    '<svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">'
    '<path class="aeon-icon-pause" d="M4 3h3v10H4zM9 3h3v10H9z"/>'
    '<path class="aeon-icon-play" d="M4.5 2.5v11l9-5.5z"/></svg>'
    "<span></span></button></div>"
)
html.append("</div>")

out = Path(__file__).parent / "hero.html"
out.write_text("\n".join(html) + "\n", encoding="utf-8")
