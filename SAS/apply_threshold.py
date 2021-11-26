import pandas as pd
import numpy as np


def apply_threshhold_to_df(new_df, threshhold_epsilon):
    # Load and transform data
    original_df = pd.read_csv('transform_dataset_to_time_series_student_activity_withLA_withScore_v8.csv')
    original_column_names = new_df.columns
    time_series_feature_length = 11
    time_series_length = 21
    num_fake_samples = len(new_df)
    num_original_samples = len(original_df)
    new_df_la_total_counts = [time_series_length * num_fake_samples for _ in range(time_series_feature_length)]
    original_df_la_total_counts = [time_series_length * num_original_samples for _ in range(time_series_feature_length)]

    original_df = original_df.drop(columns=['student_id', 'program_id', 'team_id', 'start_date', 'score_week1', 'score_week2', 'score_week3'])
    X = np.reshape(original_df.values, (num_original_samples, time_series_feature_length, time_series_length))

    new_df_vals = new_df.values
    X_hat = np.reshape(new_df_vals, (num_fake_samples, time_series_feature_length, time_series_length))

    # Function to calculate the number of zero values in a given dimension
    def calculate_zero_counts_in_feature(calc_X):
        total_zeros = 0
        for row in calc_X:
            for time_series_point in row:
                if time_series_point <= 0:
                    total_zeros += 1
        return total_zeros


    # Progressively iterate the cutoff threshold i until the number of zero values in the dimension meets the ratio
    # of zero values in the original dataset
    def iteratively_increase_threshold(X_dim, original_zero_ratio):
        # Check the case where the perturbation caused the zero ratio to be > 1
        if original_zero_ratio > 1:
            original_zero_ratio = 1
        average_val = sum(sum(X_dim)) / X_dim.size
        total_count = X_dim.size
        threshold_val = 1
        while (calculate_zero_counts_in_feature(X_dim) / total_count) < original_zero_ratio:
            threshold_function = np.vectorize(lambda x: 0 if x < threshold_val else x)
            X_dim = threshold_function(X_dim)
            threshold_val += 1
        print(f'Threshold was: {threshold_val}, Average val is: {average_val}, Ratio average / threshold is: {average_val / threshold_val}')
        return X_dim


    # Calculate the zero counts and ratios for the original data
    original_zero_counts = [calculate_zero_counts_in_feature(X[:, i, :]) for i in range(time_series_feature_length)]
    original_zero_ratios = [zero_count / la_total_count for zero_count, la_total_count in zip(original_zero_counts, original_df_la_total_counts)]
    perturbed_original_zero_ratios = original_zero_ratios + \
                                     np.random.laplace(np.zeros_like(original_zero_ratios), (1 / (len(original_df) * threshhold_epsilon)))
    # TODO: Take min value between perturbed_original_zero_ratios and a threshhold (= 1 - 1 / (len(original_df)))
    # Also add threshhold for 0 minimum
    print('Original counts, zero counts, ratios')
    print(original_df_la_total_counts)
    print(original_zero_counts)
    print(perturbed_original_zero_ratios)
    print()

    # Calculate the zero counts and ratios for the DP GAN data
    new_zero_counts = [calculate_zero_counts_in_feature(X_hat[:, i, :]) for i in range(time_series_feature_length)]
    new_zero_ratios = [zero_count / la_total_count for zero_count, la_total_count in zip(new_zero_counts, new_df_la_total_counts)]
    print('DP GAN counts, zero counts, ratios')
    print(new_df_la_total_counts)
    print(new_zero_counts)
    print(new_zero_ratios)
    print()

    # Create a new dataset with the zero cutoff progressively applied to the DP GAN data
    X_cutoff = np.zeros_like(X_hat)
    for i in range(time_series_feature_length):
        X_cutoff[:, i, :] = iteratively_increase_threshold(X_hat[:, i, :], perturbed_original_zero_ratios[i])

    output_zero_counts = [calculate_zero_counts_in_feature(X_cutoff[:, i, :]) for i in range(time_series_feature_length)]
    output_zero_ratios = [zero_count / la_total_count for zero_count, la_total_count in zip(output_zero_counts, new_df_la_total_counts)]
    print('Threshold applied counts, zero counts, ratios')
    print(new_df_la_total_counts)
    print(output_zero_counts)
    print(output_zero_ratios)
    print()

    # Save new dataset
    output_dataframe = pd.concat([pd.DataFrame(X_cutoff[:, i, :]) for i in range(time_series_feature_length)], axis=1)
    output_dataframe_with_scores = pd.concat([output_dataframe], axis=1)
    output_dataframe_with_scores.columns = original_column_names
    # output_dataframe_with_scores.to_csv('applied_threshhold_data.csv', header=True, index=False)

    return output_dataframe_with_scores