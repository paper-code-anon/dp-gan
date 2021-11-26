import pandas as pd
import numpy as np


def reverse_transform_noisy_and_apply_relu(sample, means, scales, inverse_transformation_sensitivity, N, inverse_eps):
    noisy_mean = means + \
                 np.random.laplace(np.zeros_like(means), (inverse_transformation_sensitivity / N / inverse_eps))
    noisy_scale = np.sqrt(np.abs(scales ** 2 +
                          np.random.laplace(np.zeros_like(scales), (inverse_transformation_sensitivity ** 2 / N / inverse_eps))))
    noisy_inverse = sample * noisy_scale + noisy_mean
    noisy_inverse[noisy_inverse < 0] = 0
    return noisy_inverse


def apply_private_inverse_transformation_fun(df, means_df, scales_df, inverse_transformation_epsilon):
    time_series_feature_length = 11
    time_series_length = 21
    inverse_transformation_sensitivity = 200
    N = len(df)
    original_column_names = df.columns
    X_unscaled_unshaped = np.reshape(df.values, (len(df),  time_series_feature_length, time_series_length))
    X_unscaled = np.swapaxes(X_unscaled_unshaped, 1, 2)
    means_array = means_df.values
    scales_array = scales_df.values

    saved_dim_vals = []
    for i in range(time_series_feature_length):
        X_dim = X_unscaled[:, :, i]
        means_dim = means_array[i]
        scales_dim = scales_array[i]
        inverse_relu_samples = []
        for sample in X_dim:
            inverse_relu_samples.append(reverse_transform_noisy_and_apply_relu(sample,
                                                                               means_dim,
                                                                               scales_dim,
                                                                               inverse_transformation_sensitivity,
                                                                               N,
                                                                               inverse_transformation_epsilon))
        saved_dim_vals.append(pd.DataFrame(inverse_relu_samples))
    output_dataframe = pd.concat(saved_dim_vals, axis=1)
    output_dataframe.columns = original_column_names
    return output_dataframe


