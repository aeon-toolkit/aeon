"""MultiRocketMultivariate test code."""

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aeon.datasets import load_basic_motions
from aeon.transformations.collection.convolution_based import MultiRocketMultivariate


def test_multirocket_multivariate_on_basic_motions():
    """Test of MultiRocketMultivariate on basic motions."""
    # load training data
    X_training, Y_training = load_basic_motions(split="train")

    # 'fit' MultiRocket -> infer data dimensions, generate random kernels
    multirocket = MultiRocketMultivariate(random_state=0, num_kernels=100)
    multirocket.fit(X_training)

    # transform training data
    X_training_transform = multirocket.transform(X_training)

    # test shape of transformed training data  nearest multiple of 4*84=336
    np.testing.assert_equal(X_training_transform.shape, (len(X_training), 672))

    # fit classifier
    classifier = make_pipeline(
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
    )
    classifier.fit(X_training_transform, Y_training)

    # load test data
    X_test, Y_test = load_basic_motions(split="test")

    # transform test data
    X_test_transform = multirocket.transform(X_test)

    # test shape of transformed training data  nearest multiple of 4*84=336
    np.testing.assert_equal(X_test_transform.shape, (len(X_test), 672))

    # todo: below has been temporarily commented due to inconsistency in random state

    # predict (alternatively: 'classifier.score(X_test_transform, Y_test)')
    # predictions = classifier.predict(X_test_transform)

    # accuracy = accuracy_score(predictions, Y_test)

    # test predictions (on BasicMotions, should be 100% accurate)
    # assert accuracy == 1.0
