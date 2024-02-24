import numpy as np

from aeon.transformations.series import SpectogramTransformer


def test_spectogram_transformer():
    series = np.array([584.0, -11.0, 23.0, 79.0, 1001.0, 0.0, -19.0])

    expected_sample_freq = np.array([0.0, 0.14285714, 0.28571429, 0.42857143])
    expected_segment_time = np.array([3.5])
    expected_spectrogram = np.array(
        [[20101.22789116], [238581.86979207], [439788.78395447], [278571.85645754]]
    )

    st = SpectogramTransformer()
    res1, res2, res3 = st.transform(series)
    np.testing.assert_allclose(
        np.array(
            [res1, res2, res3]),
            np.array(
                [expected_sample_freq, expected_segment_time,\
                expected_spectrogram]
            ),
        )