"""BOSS test code."""

import numpy as np
from sklearn.metrics import accuracy_score

from aeon.classification.dictionary_based import BOSSEnsemble, ContractableBOSS
from aeon.datasets import load_unit_test
from aeon.testing.utils.collection import make_2d_test_data


def test_boss_train_estimate():
    """Test of BOSS train estimate on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")

    # train BOSS
    boss = BOSSEnsemble(
        max_ensemble_size=2, random_state=0, save_train_predictions=True
    )
    boss.fit(X_train, y_train)

    # test train estimate
    train_probas = boss._get_train_probs(X_train, y_train)
    assert train_probas.shape == (20, 2)
    train_preds = boss.classes_[np.argmax(train_probas, axis=1)]
    assert accuracy_score(y_train, train_preds) >= 0.6


def test_cboss_small_train():
    """Test with a small amount of train cases, subsampling can cause issues."""
    X, y = make_2d_test_data(n_cases=3, n_timepoints=20, n_labels=2)
    cboss = ContractableBOSS()
    cboss.fit(X, y)
    cboss.predict(X)
