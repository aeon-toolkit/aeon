# The key string (i.e. 'euclidean') must be the same as the name in _registry

_expected_distance_results = {
    # Result structure:
    # [single value series, univariate series, multivariate series, dataset,
    #   unequal univariate, multivariate unequal, dataset unequal]
    "euclidean": [
        5.0,
        2.6329864895136623,
        7.093596608755006,
        2.3179478388647876,
        4.938969178714248,
    ],
    "minkowski": [
        5.0,
        2.6329864895136623,
        7.093596608755006,
        2.3179478388647876,
        4.938969178714248,
    ],
    "erp": [
        5.0,
        5.037672883414786,
        20.724482073800456,
        5.16666987535642,
        18.353616712062177,
    ],
    "edr": [1.0, 0.6, 1.0, 0.4, 0.5],
    "lcss": [1.0, 0.1, 1.0, 0.0, 1.0],
    "squared": [
        25.0,
        6.932617853961479,
        50.31911284774053,
        5.37288218369794,
        24.39341654828929,
    ],
    "dtw": [
        25.0,
        2.180365495972097,
        47.59969618998147,
        4.360373075383168,
        44.86527164702194,
    ],
    "shape_dtw": [
        25.0,
        6.93261785396148,
        50.319112847740534,
        7.221098545890512,
        47.6105369804373,
    ],
    "ddtw": [
        0.0,
        2.0884818837222006,
        34.837800040564005,
        3.6475610211489875,
        35.916981095128804,
    ],
    "wdtw": [
        12.343758137512241,
        0.985380547171357,
        21.265839226825413,
        2.0040890166976926,
        20.795690703034445,
    ],
    "wddtw": [
        0.0,
        0.9736009365730778,
        15.926194649221529,
        1.7031094916423124,
        16.967390011736825,
    ],
    "msm": [
        5.0,
        6.828557434224288,
        54.950486942429855,
        9.155720688607646,
        83.80645242153975,
    ],
    "twe": [
        5.0,
        11.33529624872385,
        40.346435059599386,
        12.461233755089522,
        36.0265974253265,
    ],
    "adtw": [
        25.0,
        5.163699519307769,
        49.59969618998147,
        9.360373075383167,
        49.86527164702194,
    ],
    "manhattan": [
        5.0,
        7.399565602170839,
        56.41541036195164,
        4.788349892989078,
        28.544576071968976,
    ],
}

_expected_distance_results_params = {
    # Result structure:
    # [univariate series, multivariate series]
    "dtw": [
        [3.088712375990371, 47.59969618998147],
        [3.2313355792547123, 48.59583629653524],
    ],
    "erp": [
        [0.6648081862148058, 4.365472428062562],
        [0.12468518773870504, 1.2934475102361824],
        [5.279748833082764, 20.71830043391765],
        [7.399565602170839, 21.90092421200079],
    ],
    "edr": [
        [0.3, 0.3],
        [0.1, 0.1],
        [0.3, 1.0],
    ],
    "lcss": [
        [0.09999999999999998, 1.0],
        [0.19999999999999996, 1.0],
        [0.30000000000000004, 1.0],
    ],
    "ddtw": [
        [2.683958998434711, 34.837800040564005],
        [6.107450358980301, 36.86313484277111],
    ],
    "wdtw": [
        [1.364177898415523, 21.265839226825413],
        [1.4215792172731838, 21.526286405547953],
        [0.02752598656586074, 0.33677832093219406],
    ],
    "wddtw": [
        [1.223806382386346, 15.926194649221529],
        [2.749366514714634, 16.59453005870139],
        [0.08524969290653987, 0.663028083142974],
    ],
    "twe": [
        [4.045224987722827, 8.724529215084788],
        [1.0754723019677446, 3.1555357525791856],
        [8.892626757586168, 39.346435059599386],
        [14.469312584616958, 41.0046466591961],
    ],
    "msm": [
        [3.4979588555649106, 36.44446666081217],
        [1.0, 11.330243154019655],
        [6.5373486105638134, 50.88045781961115],
        [3.6922226886554435, 38.923480043064245],
    ],
    "adtw": [
        [5.163699519307769, 49.59969618998147],
        [5.231335579254712, 50.31911284774053],
        [6.932617853961479, 50.31911284774053],
    ],
}
