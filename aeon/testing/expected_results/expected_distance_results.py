"""Expected results for distances."""

# The key string (i.e. 'euclidean') must be the same as the name in _registry

_expected_distance_results = {
    # Result structure:
    # [single value series, univariate series, multivariate series, dataset,
    #   unequal univariate, multivariate unequal, dataset unequal]
    "euclidean": [
        5.0,
        5.45622611835132,
        14.285548782488558,
        4.547634701488958,
        9.938223910720454,
    ],
    "squared": [
        25.0,
        29.770403454579107,
        204.07690401686034,
        20.680981378186566,
        98.76829449961575,
    ],
    "manhattan": [
        5.0,
        15.460501609188993,
        108.935882528437,
        8.513472312465874,
        53.5752848256043,
    ],
    "minkowski": [
        5.0,
        5.45622611835132,
        14.285548782488558,
        4.547634701488958,
        9.938223910720454,
    ],
    "dtw": [
        5.0,
        8.559917930965497,
        44.3099600330504,
        9.594320378741982,
        29.148827543490025,
    ],
    "ddtw": [
        0.0,
        8.353927534888799,
        132.73682936639952,
        14.590244084595948,
        68.49854588751192,
    ],
    "wdtw": [
        12.343758137512241,
        4.901736795260025,
        82.36559669136193,
        5.000093573483088,
        43.243080299363754,
    ],
    "wddtw": [
        0.0,
        3.8944037462923102,
        61.091183129252485,
        6.812437966569249,
        30.83571659208905,
    ],
    "lcss": [1.0, 0.30000000000000004, 1.0, 0.0, 0.9],
    "erp": [
        5.0,
        13.782010409379064,
        44.3099600330504,
        16.362513700113617,
        29.148827543490025,
    ],
    "edr": [1.0, 0.6, 1.0, 0.4, 1.0],
    "twe": [
        5.0,
        18.24328131880782,
        78.81976840746147,
        19.11304326116219,
        54.77676226769671,
    ],
    "msm": [
        5.0,
        12.099213975730216,
        92.75733240032741,
        12.405615515950716,
        47.01212744615127,
    ],
    "adtw": [
        25.0,
        16.098799155006947,
        186.1829978640948,
        20.01992239481733,
        98.76829449961573,
    ],
    "shape_dtw": [
        25.0,
        29.770403454579107,
        204.0769040168604,
        15.359727902635804,
        98.76829449961573,
    ],
    "sbd": [
        0.0,
        0.33449275676180856,
        0.5108776877887851,
        0.5967483689427071,
        0.6010116433765647,
    ],
}

_expected_distance_results_params = {
    # Result structure:
    # [univariate series, multivariate series]
    "dtw": [
        [10.441974486203806, 44.3099600330504],
        [10.8024655063829, 44.3099600330504],
    ],
    "ddtw": [
        [10.735835993738842, 132.73682936639955],
        [24.429801435921203, 151.16013640991326],
    ],
    "wdtw": [
        [6.127263888571422, 82.36559669136193],
        [6.26486870580251, 87.68847937158377],
        [0.11800092225918553, 1.3658562956744358],
    ],
    "wddtw": [
        [4.895225529545384, 61.09118312925249],
        [10.997466058858535, 68.04715437335123],
        [0.3409987716261595, 2.7187979513671015],
    ],
    "lcss": [[0.30000000000000004, 1.0], [0.4, 1.0], [0.4, 1.0]],
    "edr": [[0.3, 0.3], [0.1, 0.1], [0.5, 1.0]],
    "twe": [
        [5.087449975445656, 15.161815735222117],
        [1.1499446039354893, 5.995665808293953],
        [15.243281318807819, 77.81976840746147],
        [27.97089924329228, 83.97624505343292],
    ],
    "msm": [
        [4.080245996952201, 43.583053575960584],
        [1.0, 15.829914369482566],
        [12.023580258367444, 88.80013932627139],
        [7.115130579734542, 61.80633627614831],
    ],
    "adtw": [
        [16.098799155006947, 186.18299786409477],
        [16.25013434881487, 199.8788694266691],
        [24.098799155006947, 194.18299786409477],
    ],
    "sbd": [[0.13378563362841267, 0.12052110294129567]],
    "erp": [
        [6.1963403666089425, 23.958805888780923],
        [2.2271884807416047, 9.205416143392629],
        [12.782010409379064, 44.3099600330504],
        [15.460501609188993, 44.3099600330504],
    ],
    "minkowski": [
        [15.460501609188993, 108.935882528437],
        [5.45622611835132, 14.285548782488558],
    ],
    "shape_dtw": [
        [29.770403454579107, 204.0769040168604],
        [29.770403454579107, 204.0769040168604],
        [20.36273099843225, 187.37458448152938],
        [20.36273099843225, 187.37458448152938],
    ],
}
