"""STC specific tests."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from aeon.classification.shapelet_based import ShapeletTransformClassifier
from aeon.classification.sklearn import RotationForestClassifier
from aeon.testing.data_generation import make_example_3d_numpy


def test_predict_proba():
    """Test predict_proba when classifier has no predict_proba method."""
    X = make_example_3d_numpy(return_y=False, n_cases=10)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    stc = ShapeletTransformClassifier(estimator=SVC(probability=False))
    stc.fit(X, y)
    probas = stc._predict_proba(X)
    assert np.all(
        (probas == 0.0) | (probas == 1.0)
    ), "Array contains values other than 0 and 1"
    with pytest.raises(ValueError, match="Estimator must have a predict_proba method"):
        stc._fit_predict_proba(X, y)
    stc = ShapeletTransformClassifier(estimator=RandomForestClassifier(n_estimators=10))
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    with pytest.raises(ValueError, match="All classes must have at least 2 values"):
        stc._fit_predict_proba(X, y)


def test_predict_proba_unequal_length_list():
    """Test predict_proba on unequal-length list input (gh#3447)."""
    import numpy as np
    from sklearn.svm import LinearSVC

    from aeon.classification.shapelet_based import ShapeletTransformClassifier

    X = [np.random.RandomState(i).rand(1, 8 + i) for i in range(6)]
    y = np.array([0, 1, 0, 1, 0, 1])

    clf = ShapeletTransformClassifier(estimator=LinearSVC(), random_state=0)
    clf.fit(X, y)
    proba = clf.predict_proba(X[:2])
    assert proba.shape == (2, clf.n_classes_)


@pytest.mark.parametrize("verbose", [0, 1, 2])
def test_stc_verbosity_levels(verbose, capsys):
    """STC reports phases and propagates detailed component verbosity."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_timepoints=12,
        n_labels=2,
        random_state=0,
    )
    stc = ShapeletTransformClassifier(
        estimator=RotationForestClassifier(n_estimators=2),
        n_shapelet_samples=20,
        max_shapelets=4,
        batch_size=10,
        random_state=0,
        verbose=verbose,
    )

    stc.fit(X, y)
    fit_output = capsys.readouterr().out

    assert stc._transformer.verbose == verbose
    assert stc._estimator.verbose == verbose
    if verbose == 0:
        assert fit_output == ""
    else:
        assert "[STC] Starting fit: n_cases=12" in fit_output
        assert "[STC] Finished shapelet transform in " in fit_output
        assert "[STC] Finished estimator fit in " in fit_output
        assert "[STC] Finished fit in " in fit_output
        if verbose == 1:
            assert "[RST] Progress: extracted=" in fit_output
            assert "[RotF] Progress: built=" in fit_output
            assert "[RST] Batch " not in fit_output
            assert "[RotF] Estimator " not in fit_output
        else:
            assert "[RST] Batch 1:" in fit_output
            assert "[RotF] Estimator 1/2:" in fit_output

    stc.predict_proba(X[:2])
    predict_output = capsys.readouterr().out
    if verbose == 0:
        assert predict_output == ""
    else:
        assert "[STC] Finished transform for predict_proba in " in predict_output
        assert "[STC] Finished probability prediction in " in predict_output

    stc.predict(X[:2])
    predict_output = capsys.readouterr().out
    if verbose == 0:
        assert predict_output == ""
    else:
        assert "[STC] Finished transform for predict in " in predict_output
        assert "[STC] Finished prediction in " in predict_output


def test_stc_train_estimate_verbosity(capsys):
    """STC reports its RotationForest OOB train-estimate phase."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_timepoints=12,
        n_labels=2,
        random_state=0,
    )
    stc = ShapeletTransformClassifier(
        estimator=RotationForestClassifier(n_estimators=2),
        n_shapelet_samples=10,
        max_shapelets=4,
        batch_size=5,
        random_state=0,
        verbose=1,
    )

    proba = stc.fit_predict_proba(X, y)
    output = capsys.readouterr().out

    assert proba.shape == (len(y), 2)
    assert (
        "[STC] Starting estimator fit and train estimates (RotationForest OOB)"
        in output
    )
    assert "[RotF] Starting fit:" in output
    assert "[STC] Finished estimator fit and train estimates in " in output


def test_stc_contract_verbosity_is_propagated(capsys):
    """STC passes its allocated contract and verbosity into both components."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_timepoints=12,
        n_labels=2,
        random_state=0,
    )
    stc = ShapeletTransformClassifier(
        estimator=RotationForestClassifier(contract_max_n_estimators=1),
        time_limit_in_minutes=0.03,
        contract_max_n_shapelet_samples=5,
        max_shapelets=4,
        batch_size=5,
        random_state=0,
        verbose=2,
    )

    stc.fit(X, y)
    output = capsys.readouterr().out

    assert stc._transform_limit_in_minutes == pytest.approx(0.016)
    assert stc._estimator.time_limit_in_minutes == pytest.approx(0.01)
    assert "[RST] Starting fit: mode=contract" in output
    assert "[RST] Batch 1:" in output
    assert "[RotF] Estimator 1:" in output
