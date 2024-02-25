import numpy as np

from aeon.transformations.series import SpectrogramTransformer


def test_spectogram_transformer():
    series = np.array([584.0, -11.0, 23.0, 79.0, 1001.0, 0.0, -19.0])

    expected_sample_freq = np.array([0.0, 0.14285714, 0.28571429, 0.42857143])
    expected_segment_time = np.array([3.5])
    expected_spectrogram = np.array(
        [[20101.22789116], [238581.86979207], [439788.78395447], [278571.85645754]]
    )

    st = SpectrogramTransformer()
    res1, res2, res3 = st.fit_transform(series)
    np.testing.assert_allclose(res1, expected_sample_freq)
    np.testing.assert_allclose(res2, expected_segment_time)
    np.testing.assert_allclose(res3, expected_spectrogram)
