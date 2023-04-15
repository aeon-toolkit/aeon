# import aeon
# from aeon.distance_rework.test._utils import create_test_distance_numpy, _time_distance
# import numpy as np
#
#
# def _test_compare(orginal, new):
#     x = create_test_distance_numpy(10, 1, 10, 1)
#     y = create_test_distance_numpy(10, 1, 10, 2)
#     print("+++++++++++++++++++++++++++++")
#     for i in range(10):
#         curr_x = x[i]
#         curr_y = y[i]
#         print("original", orginal(curr_x, curr_y))
#         print("new", new(curr_x, curr_y))
#         print(orginal(curr_x, curr_y) == new(curr_x, curr_y))
#         # np.testing.assert_almost_equal(orginal(curr_x, curr_y), new(curr_x, curr_y))
#
# def _test_speed_time_points(orginal, new, num_timepoints, metric_name, df):
#     new_speed = []
#     old_speed = []
#
#     for num_timepoint in num_timepoints:
#         x = create_test_distance_numpy(2, num_timepoint, 100, 1)[0]
#         y = create_test_distance_numpy(2, num_timepoint, 100, 2)[0]
#         kwargs = {"x": x, "y": y}
#         old_speed.append(_time_distance(orginal, average=10, **kwargs))
#         new_speed.append(_time_distance(new, average=10, **kwargs))
#     df.loc[f"{metric_name}_old"] = old_speed
#     df.loc[f"{metric_name}_new"] = new_speed
#


# def test_compare():
#     _test_compare(aeon.distances.euclidean_distance, aeon.distance_rework.euclidean_distance)
#     _test_compare(aeon.distances.squared_distance, aeon.distance_rework.squared_distance)
#     _test_compare(aeon.distances.dtw_distance, aeon.distance_rework.dtw_distance)
#     _test_compare(aeon.distances.ddtw_distance, aeon.distance_rework.ddtw_distance)
#     _test_compare(aeon.distances.lcss_distance, aeon.distance_rework.lcss_distance)
#     # _test_compare(aeon.distances.edr_distance, aeon.distance_rework.edr_distance)
#     # _test_compare(aeon.distances.erp_distance, aeon.distance_rework.erp_distance)
#     # _test_compare(aeon.distances.wddtw_distance, aeon.distance_rework.wddtw_distance)
#     _test_compare(aeon.distances.wdtw_distance, aeon.distance_rework.wdtw_distance)
#     _test_compare(aeon.distances.msm_distance, aeon.distance_rework.msm_distance)
#     # _test_compare(aeon.distances.twe_distance, aeon.distance_rework.twe_distance)
#
#
# def test_not_equal_compare():
#     print("edr")
#     _test_compare(aeon.distances.edr_distance, aeon.distance_rework.edr_distance)
#     print("erp")
#     _test_compare(aeon.distances.erp_distance, aeon.distance_rework.erp_distance)
#     print("wddtw")
#     _test_compare(aeon.distances.wddtw_distance, aeon.distance_rework.wddtw_distance)
#     print("twe")
#     _test_compare(aeon.distances.twe_distance, aeon.distance_rework.twe_distance)
#
# import pandas as pd
#
#
#
# def test_speed():
#     num_timepoints = [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
#     result_df = pd.DataFrame(columns=num_timepoints)
#     _test_speed_time_points(aeon.distances.euclidean_distance, aeon.distance_rework.euclidean_distance, num_timepoints, "euclidean", result_df)
#     _test_speed_time_points(aeon.distances.squared_distance, aeon.distance_rework.squared_distance, num_timepoints, "squared", result_df)
#     _test_speed_time_points(aeon.distances.dtw_distance, aeon.distance_rework.dtw_distance, num_timepoints, "dtw", result_df)
#     _test_speed_time_points(aeon.distances.ddtw_distance, aeon.distance_rework.ddtw_distance, num_timepoints, "ddtw", result_df)
#     _test_speed_time_points(aeon.distances.lcss_distance, aeon.distance_rework.lcss_distance, num_timepoints, "lcss", result_df)
#     _test_speed_time_points(aeon.distances.edr_distance, aeon.distance_rework.edr_distance, num_timepoints, "edr", result_df)
#     _test_speed_time_points(aeon.distances.erp_distance, aeon.distance_rework.erp_distance, num_timepoints, "erp", result_df)
#     _test_speed_time_points(aeon.distances.wddtw_distance, aeon.distance_rework.wddtw_distance, num_timepoints, "wddtw", result_df)
#     _test_speed_time_points(aeon.distances.wdtw_distance, aeon.distance_rework.wdtw_distance, num_timepoints, "wdtw", result_df)
#     # _test_speed_time_points(aeon.distances.msm_distance, aeon.distance_rework.msm_distance, num_timepoints, "msm", result_df)
#     _test_speed_time_points(aeon.distances.twe_distance, aeon.distance_rework.twe_distance, num_timepoints, "twe", result_df)
#
#     result_df.to_csv("speed_dims.csv", index=True)

#
# def test_read():
#     distances = ["euclidean", "squared", "dtw", "ddtw", "lcss", "edr", "erp", "wddtw", "wdtw", "msm", "twe"]
#     df = pd.read_csv("speed.csv")
#
#     for distance in distances:
#         print("+++++")
#         old = df.loc[f"{distance}_old"]
#         new = df.loc[f"{distance}_new"]
#         print(distance)
#         print(old)
#         print(new)
#         print(old/new)
#     print(df)
