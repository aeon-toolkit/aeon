import numpy as np

from aeon.transformations.collection.contrastive_based._ts2vec import TS2Vec



def test_shape():
    expected_features = 200
    X = np.random.random(size=(10, 1, 100))
    transformer = TS2Vec(output_dim=expected_features)
    transformer.fit(X)
    X_trans = transformer.transform(X)
    np.testing.assert_equal(X_trans.shape, (len(X), expected_features))

def test_shape2():
    expected_features = 500
    X = np.random.random(size=(10, 1, 100))
    transformer = TS2Vec(output_dim=expected_features)
    transformer.fit(X)
    X_trans = transformer.transform(X)
    np.testing.assert_equal(X_trans.shape, (len(X), expected_features))

def test_shape3():
    expected_features = 200
    X = np.random.random(size=(10, 3, 100))
    transformer = TS2Vec(output_dim=expected_features)
    transformer.fit(X)
    X_trans = transformer.transform(X)
    np.testing.assert_equal(X_trans.shape, (len(X), expected_features))