"""Tests for the MrSQM classifier.

Expected values were produced by the original ``mrsqm`` 0.0.7 C++ implementation
(https://github.com/mlgig/mrsqm) using the same inputs.
"""

import numpy as np
import pytest

from aeon.classification.dictionary_based import MrSQMClassifier
from aeon.classification.dictionary_based._mrsqm import (
    _normalise_rows,
    _sax_breakpoints,
    _sax_words,
    _sfa_lookup_table,
    _sfa_words,
    _sqm_mine,
)
from aeon.testing.data_generation import make_example_3d_numpy


def _join(words, n_words, i):
    return b" ".join(words[i, j].tobytes() for j in range(n_words[i]))


def _sfa(X, window, word, alphabet):
    Xn = _normalise_rows(X, True)
    table = _sfa_lookup_table(Xn, window, word + 1, alphabet, False, 0)
    dft = np.fft.rfft(Xn[:, :window], axis=1)
    words, n_words = _sfa_words(
        Xn,
        window,
        word + 1,
        alphabet,
        0,
        table,
        np.ascontiguousarray(dft.real),
        np.ascontiguousarray(dft.imag),
    )
    return table, words, n_words


@pytest.fixture
def data():
    """Small univariate dataset used to generate the original's outputs."""
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=1, n_timepoints=40, random_state=0
    )
    return X, y


def test_sax_breakpoints_match_original():
    """Breakpoints are the truncated values hard coded in the original."""
    np.testing.assert_array_equal(_sax_breakpoints(2), [0.0])
    np.testing.assert_array_equal(
        _sax_breakpoints(5),
        [-0.841621233573, -0.253347103136, 0.253347103136, 0.841621233573],
    )
    np.testing.assert_array_equal(
        _sax_breakpoints(7),
        [
            -1.06757052388,
            -0.565948821933,
            -0.180012369793,
            0.180012369793,
            0.565948821933,
            1.06757052388,
        ],
    )
    assert _sax_breakpoints(13)[5] == -0.0965586152896
    assert _sax_breakpoints(16)[0] == -1.53412054435


def test_sax_words_match_original(data):
    """SAX words with numerosity reduction, with and without dilation."""
    X = data[0][:, 0, :]
    words, n_words = _sax_words(X, 16, 4, _sax_breakpoints(3), 1)
    assert _join(words, n_words, 0) == (
        b"bbcb bcba bbca bcbb bbab bbac babc bacb bbcb abcb acbb bcbb cbba cbbb "
        b"cabb cbbb bbbb bbbc"
    )
    words, n_words = _sax_words(X, 8, 5, _sax_breakpoints(6), 2)
    assert _join(words, n_words, 1) == (
        b"ddaec decdb ebcec ddebc cbfcd edcbc bdedb cfbdb cfccc fcccb ecdbb ebdbe "
        b"edbcd becdd bcccf debfb cabef dbdbe bbcef cbccf abeee bdbfd bceeb ccdfb "
        b"befda dbfbb"
    )


def test_sfa_matches_original(data):
    """SFA lookup table and words."""
    X = data[0][:, 0, :]
    table, words, n_words = _sfa(X, 16, 7, 4)
    expected_table = np.array(
        [
            [-0.76, -0.07, 0.42],
            [-0.0, np.inf, np.inf],
            [-0.81, -0.24, 0.49],
            [-0.25, 0.17, 0.97],
            [-0.66, 0.05, 0.75],
            [-0.3, 0.16, 0.54],
            [-0.55, -0.19, 0.28],
            [-0.31, 0.11, 0.77],
        ]
    )
    np.testing.assert_array_equal(table, expected_table)
    assert _join(words, n_words, 0) == (
        b'"%*.35< "%+/48< "&,/389 #\',/189 #(,/159 $(+.15; $(*-17< $()-28; '
        b'$()-489 #\').469 #&)045: "%)046< #%)018; #%*/189 "%+-169 "&,-15; '
        b"#',-37; $(,.489 $(+/46: \"(+/45< \"'+038< !'*028; !'*/189 \"(*/169 "
        b"\"').15:"
    )

    # first order differences with a different window and word length
    X_diff = np.diff(X, axis=1, prepend=0)
    _, words, n_words = _sfa(X_diff, 12, 8, 3)
    assert _join(words, n_words, 3) == (
        b"#&'+/246 !$'+/058 !%',.148 !$'+-046 \"%'*-048 \"%'*-048 \"%'*/146 "
        b"#&',/138 \"$',/058 #$',-156 \"$(*-148 !$)*.148 #&)+.036 #&(+.046 "
        b"#&'*.158 !$(*/256 #%(,.236 \"%(,-038 #%'+-057 !$(*.257 #%(+.238 "
        b'!$)*/046 "%(,.057 !%(+-256 "%(*-237 "%\'*.037 "%\'*/057 "%\'+/257 '
        b"!$'+/137"
    )


def test_sqm_matches_original(data):
    """Top chi-squared subsequences mined with SQM."""
    X, y = data
    _, words, n_words = _sfa(X[:, 0, :], 16, 7, 4)
    mined = _sqm_mine(words, n_words, y.astype(np.int64), 10)
    assert mined == [
        b".17;",
        b".289",
        b"16:",
        b"/16:",
        b"#',",
        b"6:",
        b"/469",
        b"/279",
        b"/25;",
        b"/16<",
    ]


def test_mrsqm_matches_original(data):
    """Configuration, selected features and predictions of the classifier."""
    X, y = data
    X_test, _ = make_example_3d_numpy(
        n_cases=10, n_channels=1, n_timepoints=40, random_state=1
    )
    clf = MrSQMClassifier(
        random_state=0, nsax=1, nsfa=1, features_per_rep=20, selection_per_rep=100
    )
    clf.fit(X, y)

    config = [
        (c["method"], c["window"], c["word"], c["alphabet"], c.get("diff", False))
        for c in clf._config
    ]
    assert config == [
        ("sax", 16, 8, 3, False),
        ("sax", 16, 7, 5, False),
        ("sax", 8, 8, 4, False),
        ("sax", 8, 8, 5, False),
        ("sax", 32, 7, 3, False),
        ("sfa", 32, 7, 2, False),
        ("sfa", 16, 7, 5, False),
        ("sfa", 16, 8, 3, True),
        ("sfa", 32, 8, 2, False),
        ("sfa", 16, 8, 4, False),
    ]
    assert [len(f) for f in clf._features] == [20] * 10
    assert clf._features[0][:8] == [
        b"bbccc",
        b"cabba",
        b"bcbaba",
        b"cccc",
        b"bbbb",
        b"bcbac",
        b"baaccbcb",
        b"bbcacbb",
    ]
    assert clf._features[-1][:8] == [
        b"79>",
        b"\"'*/4",
        b"*/16:",
        b"8<>",
        b"%*/16",
        b"47<",
        b"$',.",
        b"\"'*-369",
    ]
    np.testing.assert_array_equal(clf.predict(X_test), [0, 1, 1, 0, 0, 0, 0, 0, 0, 1])
    np.testing.assert_allclose(
        clf.predict_proba(X_test)[:, 1],
        [0.1991, 0.7944, 0.9095, 0.0309, 0.4916, 0.3666, 0.2848, 0.126, 0.2411, 0.6773],
        atol=1e-4,
    )


@pytest.mark.parametrize(
    "strat, rep0, preds",
    [
        (
            "R",
            [b"bcba", b"bbbc", b"cbb", b"cdbc", b"cabcdb"],
            [1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        ),
        (
            "S",
            [b"cabbb", b"aca", b"dbbba", b"cdbd", b"cdab"],
            [0, 1, 1, 0, 1, 0, 0, 0, 0, 1],
        ),
        (
            "SR",
            [b"dabbb", b"ccabbd", b"dbbbac", b"bcbca", b"dbd"],
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        ),
    ],
)
def test_strategies_match_original(data, strat, rep0, preds):
    """Other feature selection strategies."""
    X, y = data
    X_test, _ = make_example_3d_numpy(
        n_cases=10, n_channels=1, n_timepoints=40, random_state=1
    )
    clf = MrSQMClassifier(
        random_state=1,
        strat=strat,
        nsax=1,
        nsfa=1,
        features_per_rep=10,
        selection_per_rep=50,
    )
    clf.fit(X, y)
    assert clf._features[0][:5] == rep0
    np.testing.assert_array_equal(clf.predict(X_test), preds)


def test_custom_config(data):
    """Custom configurations are copied and not modified by fit."""
    X, y = data
    config = [
        {"method": "sax", "window": 10, "word": 5, "alphabet": 4},
        {"method": "sfa", "window": 12, "word": 6, "alphabet": 3, "diff": True},
    ]
    clf = MrSQMClassifier(custom_config=config, random_state=0)
    clf.fit(X, y)
    assert len(clf._features) == 2
    assert "signature" not in config[1]
    assert clf.predict(X).shape == (20,)

    with pytest.raises(ValueError, match="alphabet size"):
        MrSQMClassifier(
            custom_config=[{"method": "sax", "window": 10, "word": 5, "alphabet": 20}]
        ).fit(X, y)
    with pytest.raises(ValueError, match="longer than the series"):
        MrSQMClassifier(
            custom_config=[{"method": "sfa", "window": 50, "word": 6, "alphabet": 3}]
        ).fit(X, y)


def test_invalid_input(data):
    """Invalid strategies and series too short for any representation."""
    X, y = data
    with pytest.raises(ValueError, match="strat must be one of"):
        MrSQMClassifier(strat="X").fit(X, y)
    with pytest.raises(ValueError, match="No symbolic representations"):
        MrSQMClassifier().fit(X[:, :, :7], y)
