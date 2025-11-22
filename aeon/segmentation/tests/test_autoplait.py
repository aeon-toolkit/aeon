"""Simple AutoPlait test."""

__maintainer__ = []
__all__ = []

import numpy as np

from aeon.segmentation import AutoPlaitSegmenter


def test_autoplait_simple():
    """Test AutoPlait segmentation.

    Simple square wave.
    Check if the predicted change points match.
    Check if regimes alternate
    """
    ts = np.concatenate([np.ones(500), -np.ones(500), np.ones(500), -np.ones(500)])
    true_cps = [499, 999, 1499]

    autoplait = AutoPlaitSegmenter(seed=42)
    autoplait.fit(ts)
    found_cps = autoplait.predict(ts, axis=0)
    regimes = autoplait.get_regime_labels()

    assert len(found_cps) == len(true_cps)
    assert len(regimes) == len(true_cps) + 1

    for i in range(len(found_cps)):
        assert abs(found_cps[i] - true_cps[i]) <= ts.shape[0] * 0.02

    assert all(regimes[i] != regimes[i + 1] for i in range(len(regimes) - 1))


def test_autoplait_complex():
    """Test AutoPlait segmentation.

    Out of phase waves.
    Check if the predicted change points match.
    Check if regimes alternate
    """
    x = np.linspace(0, 2 * np.pi, 2000)
    wave = np.cos(x)
    ts = np.stack((wave, -wave), axis=-1)
    true_cps = [250, 375, 550, 700, 1300, 1450, 1625, 1750]

    autoplait = AutoPlaitSegmenter(seed=42)
    autoplait.fit(ts)
    found_cps = autoplait.predict(ts, axis=0)
    regimes = autoplait.get_regime_labels()

    assert len(found_cps) >= len(true_cps)
    assert len(found_cps) <= len(true_cps) + 2

    first_regime = regimes[0]
    last_regime = regimes[-1]
    assert first_regime == last_regime
