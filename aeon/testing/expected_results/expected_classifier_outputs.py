"""Dictionaries of expected outputs of classifier predict runs."""

import numpy as np

# predict_proba results on unit test data
unit_test_proba = dict()

# predict_proba results on basic motions data
basic_motions_proba = dict()


unit_test_proba["ClassifierPipeline"] = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]
)
unit_test_proba["MUSE"] = np.array(
    [
        [0.4451, 0.5549],
        [0.6613, 0.3387],
        [0.1073, 0.8927],
        [0.9414, 0.0586],
        [0.5422, 0.4578],
        [0.891, 0.109],
        [0.8057, 0.1943],
        [0.1495, 0.8505],
        [0.9554, 0.0446],
        [0.9474, 0.0526],
    ]
)
unit_test_proba["BOSSEnsemble"] = np.array(
    [
        [0.0, 1.0],
        [0.6667, 0.3333],
        [0.0, 1.0],
        [0.6667, 0.3333],
        [0.6667, 0.3333],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.6667, 0.3333],
    ]
)
unit_test_proba["ContractableBOSS"] = np.array(
    [
        [0.215, 0.785],
        [0.5312, 0.4688],
        [0.3787, 0.6213],
        [1.0, 0.0],
        [0.8475, 0.1525],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0625, 0.9375],
        [0.3787, 0.6213],
        [1.0, 0.0],
    ]
)
unit_test_proba["TemporalDictionaryEnsemble"] = np.array(
    [
        [0.3307, 0.6693],
        [0.6693, 0.3307],
        [0.0, 1.0],
        [0.5538, 0.4462],
        [0.6693, 0.3307],
        [1.0, 0.0],
        [0.4462, 0.5538],
        [0.0, 1.0],
        [0.5538, 0.4462],
        [0.4462, 0.5538],
    ]
)
unit_test_proba["WEASEL"] = np.array(
    [
        [0.2654, 0.7346],
        [0.8813, 0.1187],
        [0.1571, 0.8429],
        [0.9134, 0.0866],
        [0.5676, 0.4324],
        [0.8183, 0.1817],
        [0.667, 0.333],
        [0.0549, 0.9451],
        [0.9517, 0.0483],
        [0.8607, 0.1393],
    ]
)
unit_test_proba["WEASEL_V2"] = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]
)
unit_test_proba["REDCOMETS"] = np.array(
    [
        [0.0, 1.0],
        [0.7619, 0.2381],
        [0.0, 1.0],
        [0.7619, 0.2381],
        [0.3571, 0.6429],
        [0.881, 0.119],
        [0.7619, 0.2381],
        [0.7619, 0.2381],
        [0.881, 0.119],
        [1.0, 0.0],
    ]
)
unit_test_proba["ElasticEnsemble"] = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.6667, 0.3333],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.6667, 0.3333],
        [1.0, 0.0],
    ]
)
unit_test_proba["KNeighborsTimeSeriesClassifier"] = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]
)
unit_test_proba["Catch22Classifier"] = np.array(
    [
        [0.2, 0.8],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.7, 0.3],
        [0.7, 0.3],
        [0.9, 0.1],
        [0.7, 0.3],
        [0.1, 0.9],
        [0.7, 0.3],
        [0.9, 0.1],
    ]
)
unit_test_proba["FreshPRINCEClassifier"] = np.array(
    [
        [0.2, 0.8],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]
)
unit_test_proba["RandomIntervalClassifier"] = np.array(
    [
        [0.0, 1.0],
        [0.7, 0.3],
        [0.5, 0.5],
        [0.8, 0.2],
        [0.1, 0.9],
        [0.8, 0.2],
        [0.1, 0.9],
        [0.1, 0.9],
        [0.0, 1.0],
        [0.8, 0.2],
    ]
)
unit_test_proba["QUANTClassifier"] = np.array(
    [
        [0.095, 0.905],
        [0.905, 0.095],
        [0.01, 0.99],
        [0.955, 0.045],
        [0.92, 0.08],
        [0.98, 0.02],
        [0.83, 0.17],
        [0.26, 0.74],
        [0.82, 0.18],
        [0.945, 0.055],
    ]
)
unit_test_proba["SignatureClassifier"] = np.array(
    [
        [0.1, 0.9],
        [0.9, 0.1],
        [0.1, 0.9],
        [0.8, 0.2],
        [1.0, 0.0],
        [0.7, 0.3],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.9, 0.1],
        [1.0, 0.0],
    ]
)
unit_test_proba["SummaryClassifier"] = np.array(
    [
        [0.1, 0.9],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.7, 0.3],
        [0.8, 0.2],
        [1.0, 0.0],
    ]
)
unit_test_proba["TSFreshClassifier"] = np.array(
    [
        [0.2, 0.8],
        [0.8, 0.2],
        [0.2, 0.8],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.9, 0.1],
        [0.8, 0.2],
        [1.0, 0.0],
    ]
)
unit_test_proba["HIVECOTEV1"] = np.array(
    [
        [0.0631, 0.9369],
        [0.2612, 0.7388],
        [0.0813, 0.9187],
        [1.0, 0.0],
        [0.9503, 0.0497],
        [1.0, 0.0],
        [0.7661, 0.2339],
        [0.0134, 0.9866],
        [0.6256, 0.3744],
        [0.9234, 0.0766],
    ]
)
unit_test_proba["HIVECOTEV2"] = np.array(
    [
        [0.2239, 0.7761],
        [0.6732, 0.3268],
        [0.1211, 0.8789],
        [1.0, 0.0],
        [0.9818, 0.0182],
        [1.0, 0.0],
        [0.7201, 0.2799],
        [0.2058, 0.7942],
        [0.8412, 0.1588],
        [0.9441, 0.0559],
    ]
)
unit_test_proba["CanonicalIntervalForestClassifier"] = np.array(
    [
        [0.3, 0.7],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.9, 0.1],
        [0.7, 0.3],
        [0.9, 0.1],
        [0.5, 0.5],
        [0.3, 0.7],
        [0.5, 0.5],
        [0.8, 0.2],
    ]
)
unit_test_proba["DrCIFClassifier"] = np.array(
    [
        [0.1, 0.9],
        [0.8, 0.2],
        [0.1, 0.9],
        [1.0, 0.0],
        [0.7, 0.3],
        [0.9, 0.1],
        [0.8, 0.2],
        [0.5, 0.5],
        [0.9, 0.1],
        [1.0, 0.0],
    ]
)
unit_test_proba["IntervalForestClassifier"] = np.array(
    [
        [0.2, 0.8],
        [0.7, 0.3],
        [0.1, 0.9],
        [0.9, 0.1],
        [0.7, 0.3],
        [0.8, 0.2],
        [0.7, 0.3],
        [0.5, 0.5],
        [0.7, 0.3],
        [1.0, 0.0],
    ]
)
unit_test_proba["RandomIntervalSpectralEnsembleClassifier"] = np.array(
    [
        [0.3, 0.7],
        [0.7, 0.3],
        [0.1, 0.9],
        [0.7, 0.3],
        [0.6, 0.4],
        [0.9, 0.1],
        [0.8, 0.2],
        [0.3, 0.7],
        [0.7, 0.3],
        [0.9, 0.1],
    ]
)
unit_test_proba["RSTSF"] = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.5],
        [1.0, 0.0],
        [1.0, 0.0],
    ]
)
unit_test_proba["SupervisedTimeSeriesForest"] = np.array(
    [
        [0.0, 1.0],
        [0.9, 0.1],
        [0.1, 0.9],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.1, 0.9],
        [1.0, 0.0],
        [1.0, 0.0],
    ]
)
unit_test_proba["TimeSeriesForestClassifier"] = np.array(
    [
        [0.0, 1.0],
        [0.8, 0.2],
        [0.1, 0.9],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.8, 0.2],
        [0.3, 0.7],
        [0.8, 0.2],
        [0.9, 0.1],
    ]
)
unit_test_proba["OrdinalTDE"] = np.array(
    [
        [0.0, 1.0],
        [0.3505, 0.6495],
        [0.1753, 0.8247],
        [0.8247, 0.1753],
        [0.6495, 0.3505],
        [0.701, 0.299],
        [0.6495, 0.3505],
        [0.1753, 0.8247],
        [0.8247, 0.1753],
        [1.0, 0.0],
    ]
)
unit_test_proba["SASTClassifier"] = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]
)
unit_test_proba["Arsenal"] = np.array(
    [
        [-0.0, 1.0],
        [0.9226, 0.0774],
        [-0.0, 1.0],
        [1.0, -0.0],
        [1.0, -0.0],
        [1.0, -0.0],
        [0.9226, 0.0774],
        [-0.0, 1.0],
        [1.0, -0.0],
        [1.0, -0.0],
    ]
)
unit_test_proba["RocketClassifier"] = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]
)
unit_test_proba["HydraClassifier"] = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]
)
unit_test_proba["MultiRocketHydraClassifier"] = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]
)
unit_test_proba["ShapeletTransformClassifier"] = np.array(
    [
        [0.4, 0.6],
        [0.2, 0.8],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.8, 0.2],
        [1.0, 0.0],
        [0.8, 0.2],
        [0.0, 1.0],
        [0.8, 0.2],
        [0.6, 0.4],
    ]
)
unit_test_proba["RDSTClassifier"] = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]
)
unit_test_proba["TEASER"] = np.array(
    [
        [0.2, 0.8],
        [0.9, 0.1],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.2, 0.8],
        [0.8, 0.2],
        [1.0, 0.0],
    ]
)
unit_test_proba["ProbabilityThresholdEarlyClassifier"] = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]
)

basic_motions_proba["ClassifierChannelEnsemble"] = np.array(
    [
        [0.0, 0.0, 0.25, 0.75],
        [0.5, 0.25, 0.0, 0.25],
        [0.0, 0.25, 0.25, 0.5],
        [0.0, 0.75, 0.25, 0.0],
        [0.0, 0.25, 0.0, 0.75],
        [0.0, 0.0, 0.25, 0.75],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.5, 0.25, 0.25],
        [0.0, 0.75, 0.0, 0.25],
        [0.0, 0.75, 0.0, 0.25],
    ]
)
basic_motions_proba["ClassifierPipeline"] = np.array(
    [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
)
basic_motions_proba["MUSE"] = np.array(
    [
        [3.67057592e-05, 1.12259557e-03, 6.67246229e-04, 9.98173452e-01],
        [9.93229455e-01, 1.92232324e-04, 2.56248688e-03, 4.01582536e-03],
        [1.73244986e-04, 1.87190456e-04, 9.97716736e-01, 1.92282859e-03],
        [2.59659365e-03, 9.97076299e-01, 7.09934439e-05, 2.56113573e-04],
        [3.19356238e-05, 6.60136189e-03, 2.33211388e-03, 9.91034589e-01],
        [8.50903584e-05, 5.96209341e-04, 3.18223960e-02, 9.67496304e-01],
        [9.81362825e-01, 1.39771640e-03, 1.18616691e-02, 5.37778988e-03],
        [1.55494301e-03, 2.12773041e-04, 9.96621925e-01, 1.61035863e-03],
        [9.59903116e-03, 9.90085747e-01, 7.30870932e-05, 2.42134656e-04],
        [6.40967171e-04, 9.99163067e-01, 5.53240474e-05, 1.40642181e-04],
    ]
)
basic_motions_proba["TemporalDictionaryEnsemble"] = np.array(
    [
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.6261, 0.3739, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.7478, 0.0, 0.0, 0.2522],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.7478, 0.2522, 0.0],
        [0.0, 0.7478, 0.2522, 0.0],
    ]
)
basic_motions_proba["REDCOMETS"] = np.array(
    [
        [0.1005, 0.2374, 0.1416, 0.5205],
        [0.1461, 0.274, 0.347, 0.2329],
        [0.0548, 0.0959, 0.6119, 0.2374],
        [0.032, 0.6895, 0.0868, 0.1918],
        [0.1324, 0.2329, 0.0639, 0.5708],
        [0.1187, 0.1507, 0.1689, 0.5616],
        [0.3196, 0.2146, 0.0639, 0.4018],
        [0.0913, 0.0, 0.5845, 0.3242],
        [0.0, 0.7717, 0.0639, 0.1644],
        [0.0457, 0.79, 0.0411, 0.1233],
    ]
)
basic_motions_proba["ElasticEnsemble"] = np.array(
    [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.3333, 0.6667],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.3333, 0.6667],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
)
basic_motions_proba["KNeighborsTimeSeriesClassifier"] = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
)
basic_motions_proba["TEASER"] = np.array(
    [
        [0.0, 0.1, 0.2, 0.7],
        [0.1, 0.0, 0.3, 0.6],
        [0.0, 0.0, 1.0, 0.0],
        [0.2, 0.8, 0.0, 0.0],
        [0.0, 0.0, 0.4, 0.6],
        [0.0, 0.1, 0.2, 0.7],
        [0.5, 0.4, 0.0, 0.1],
        [0.1, 0.1, 0.6, 0.2],
        [0.0, 0.9, 0.0, 0.1],
        [0.0, 0.9, 0.0, 0.1],
    ]
)
basic_motions_proba["ProbabilityThresholdEarlyClassifier"] = np.array(
    [
        [0.1, 0.0, 0.0, 0.9],
        [0.6, 0.4, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.2, 0.8, 0.0, 0.0],
        [0.0, 0.2, 0.1, 0.7],
        [0.0, 0.1, 0.2, 0.7],
        [0.5, 0.4, 0.0, 0.1],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.9, 0.0, 0.1],
        [0.0, 0.9, 0.0, 0.1],
    ]
)
basic_motions_proba["Catch22Classifier"] = np.array(
    [
        [0.1, 0.0, 0.2, 0.7],
        [0.2, 0.3, 0.2, 0.3],
        [0.0, 0.2, 0.6, 0.2],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.1, 0.2, 0.7],
        [0.1, 0.1, 0.2, 0.6],
        [0.2, 0.5, 0.2, 0.1],
        [0.1, 0.1, 0.6, 0.2],
        [0.0, 0.7, 0.2, 0.1],
        [0.0, 1.0, 0.0, 0.0],
    ]
)
basic_motions_proba["FreshPRINCEClassifier"] = np.array(
    [
        [0.0, 0.0, 0.3, 0.7],
        [0.9, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.8, 0.2],
        [0.1, 0.9, 0.0, 0.0],
        [0.1, 0.0, 0.1, 0.8],
        [0.0, 0.0, 0.3, 0.7],
        [0.8, 0.2, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.1, 0.9, 0.0, 0.0],
        [0.1, 0.8, 0.0, 0.1],
    ]
)
basic_motions_proba["RandomIntervalClassifier"] = np.array(
    [
        [0.0, 0.0, 0.4, 0.6],
        [0.2, 0.7, 0.1, 0.0],
        [0.0, 0.0, 0.7, 0.3],
        [0.0, 0.7, 0.2, 0.1],
        [0.1, 0.0, 0.5, 0.4],
        [0.0, 0.1, 0.4, 0.5],
        [0.2, 0.7, 0.0, 0.1],
        [0.0, 0.0, 0.7, 0.3],
        [0.0, 0.5, 0.3, 0.2],
        [0.0, 0.6, 0.1, 0.3],
    ]
)
basic_motions_proba["QUANTClassifier"] = np.array(
    [
        [0.01, 0.005, 0.16, 0.825],
        [0.625, 0.22, 0.01, 0.145],
        [0.0, 0.0, 0.78, 0.22],
        [0.17, 0.74, 0.005, 0.085],
        [0.035, 0.015, 0.125, 0.825],
        [0.005, 0.0, 0.215, 0.78],
        [0.69, 0.165, 0.005, 0.14],
        [0.0, 0.0, 0.805, 0.195],
        [0.21, 0.69, 0.005, 0.095],
        [0.205, 0.7, 0.005, 0.09],
    ]
)
basic_motions_proba["SignatureClassifier"] = np.array(
    [
        [0.0, 0.0, 0.5, 0.5],
        [0.4, 0.0, 0.3, 0.3],
        [0.0, 0.0, 0.9, 0.1],
        [0.2, 0.3, 0.1, 0.4],
        [0.0, 0.0, 0.4, 0.6],
        [0.0, 0.0, 0.7, 0.3],
        [0.1, 0.0, 0.6, 0.3],
        [0.0, 0.0, 0.9, 0.1],
        [0.0, 0.7, 0.1, 0.2],
        [0.2, 0.3, 0.1, 0.4],
    ]
)
basic_motions_proba["SummaryClassifier"] = np.array(
    [
        [0.1, 0.0, 0.2, 0.7],
        [0.4, 0.2, 0.1, 0.3],
        [0.0, 0.0, 0.8, 0.2],
        [0.1, 0.9, 0.0, 0.0],
        [0.2, 0.0, 0.0, 0.8],
        [0.1, 0.0, 0.1, 0.8],
        [0.2, 0.2, 0.2, 0.4],
        [0.0, 0.0, 0.9, 0.1],
        [0.0, 1.0, 0.0, 0.0],
        [0.1, 0.9, 0.0, 0.0],
    ]
)
basic_motions_proba["TSFreshClassifier"] = np.array(
    [
        [0.0, 0.0, 0.2, 0.8],
        [0.4, 0.3, 0.0, 0.3],
        [0.0, 0.0, 0.8, 0.2],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.2, 0.2, 0.6],
        [0.0, 0.0, 0.3, 0.7],
        [0.5, 0.3, 0.0, 0.2],
        [0.0, 0.0, 0.9, 0.1],
        [0.1, 0.9, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
)
basic_motions_proba["HIVECOTEV2"] = np.array(
    [
        [0.0, 0.0, 0.0, 1.0],
        [0.7184, 0.0053, 0.2554, 0.0209],
        [0.185, 0.0851, 0.6238, 0.106],
        [0.0, 0.384, 0.2902, 0.3258],
        [0.0, 0.0, 0.1919, 0.8081],
        [0.0, 0.0, 0.0, 1.0],
        [0.7237, 0.0053, 0.0851, 0.1858],
        [0.185, 0.1866, 0.6129, 0.0155],
        [0.0053, 0.894, 0.0851, 0.0155],
        [0.0, 0.894, 0.0851, 0.0209],
    ]
)
basic_motions_proba["CanonicalIntervalForestClassifier"] = np.array(
    [
        [0.0, 0.0, 0.1, 0.9],
        [0.3, 0.4, 0.0, 0.3],
        [0.0, 0.0, 0.8, 0.2],
        [0.3, 0.3, 0.2, 0.2],
        [0.1, 0.0, 0.0, 0.9],
        [0.0, 0.0, 0.2, 0.8],
        [0.3, 0.3, 0.2, 0.2],
        [0.1, 0.2, 0.7, 0.0],
        [0.1, 0.9, 0.0, 0.0],
        [0.1, 0.7, 0.0, 0.2],
    ]
)
basic_motions_proba["RandomIntervalSpectralEnsembleClassifier"] = np.array(
    [
        [0.0, 0.1, 0.1, 0.8],
        [0.5, 0.3, 0.1, 0.1],
        [0.0, 0.0, 0.8, 0.2],
        [0.1, 0.5, 0.1, 0.3],
        [0.1, 0.0, 0.2, 0.7],
        [0.2, 0.0, 0.3, 0.5],
        [0.5, 0.4, 0.1, 0.0],
        [0.0, 0.0, 0.8, 0.2],
        [0.1, 0.6, 0.1, 0.2],
        [0.3, 0.4, 0.0, 0.3],
    ]
)
basic_motions_proba["DrCIFClassifier"] = np.array(
    [
        [0.1, 0.0, 0.2, 0.7],
        [0.5, 0.4, 0.0, 0.1],
        [0.0, 0.0, 0.8, 0.2],
        [0.1, 0.9, 0.0, 0.0],
        [0.1, 0.0, 0.3, 0.6],
        [0.0, 0.0, 0.2, 0.8],
        [0.5, 0.3, 0.0, 0.2],
        [0.0, 0.0, 0.8, 0.2],
        [0.2, 0.7, 0.0, 0.1],
        [0.0, 0.9, 0.0, 0.1],
    ]
)
basic_motions_proba["IntervalForestClassifier"] = np.array(
    [
        [0.0, 0.0, 0.3, 0.7],
        [0.6, 0.4, 0.0, 0.0],
        [0.0, 0.0, 0.9, 0.1],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.2, 0.8],
        [0.0, 0.0, 0.3, 0.7],
        [0.8, 0.2, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.1, 0.9, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
)
basic_motions_proba["SupervisedTimeSeriesForest"] = np.array(
    [
        [0.0, 0.0, 0.0, 1.0],
        [0.6, 0.2, 0.1, 0.1],
        [0.0, 0.0, 0.6, 0.4],
        [0.2, 0.8, 0.0, 0.0],
        [0.0, 0.0, 0.2, 0.8],
        [0.0, 0.1, 0.4, 0.5],
        [0.4, 0.6, 0.0, 0.0],
        [0.0, 0.0, 0.9, 0.1],
        [0.1, 0.9, 0.0, 0.0],
        [0.1, 0.9, 0.0, 0.0],
    ]
)
basic_motions_proba["TimeSeriesForestClassifier"] = np.array(
    [
        [0.0, 0.0, 0.1, 0.9],
        [0.6, 0.4, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.1, 0.9, 0.0, 0.0],
        [0.0, 0.0, 0.2, 0.8],
        [0.0, 0.0, 0.3, 0.7],
        [0.7, 0.2, 0.0, 0.1],
        [0.0, 0.0, 0.9, 0.1],
        [0.2, 0.7, 0.0, 0.1],
        [0.2, 0.7, 0.0, 0.1],
    ]
)

basic_motions_proba["RDSTClassifier"] = np.array(
    [
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
)

basic_motions_proba["Arsenal"] = np.array(
    [
        [-0.0, -0.0, -0.0, 1.0],
        [1.0, -0.0, -0.0, -0.0],
        [0.131, 0.1897, 0.4916, 0.1877],
        [0.1877, 0.131, 0.4916, 0.1897],
        [-0.0, 0.1844, 0.3775, 0.4382],
        [-0.0, -0.0, -0.0, 1.0],
        [0.8103, -0.0, -0.0, 0.1897],
        [-0.0, -0.0, 0.4949, 0.5051],
        [-0.0, 0.8156, 0.1844, -0.0],
        [-0.0, 1.0, -0.0, -0.0],
    ]
)
basic_motions_proba["RocketClassifier"] = np.array(
    [
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
)
basic_motions_proba["HydraClassifier"] = np.array(
    [
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
)
basic_motions_proba["MultiRocketHydraClassifier"] = np.array(
    [
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
)

basic_motions_proba["ShapeletTransformClassifier"] = np.array(
    [
        [0.2, 0.0, 0.0, 0.8],
        [0.0, 0.4, 0.4, 0.2],
        [0.2, 0.0, 0.4, 0.4],
        [0.2, 0.2, 0.4, 0.2],
        [0.2, 0.0, 0.0, 0.8],
        [0.2, 0.0, 0.2, 0.6],
        [0.0, 0.2, 0.4, 0.4],
        [0.0, 0.2, 0.6, 0.2],
        [0.0, 0.8, 0.2, 0.0],
        [0.0, 0.8, 0.2, 0.0],
    ]
)
