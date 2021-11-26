import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
import os

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def split_df_into_bins(df, col_name, num_bins):
    return pd.cut(df[col_name],
                  [x for x in np.arange(0, 1.001, 1 / num_bins)],
                  include_lowest=True,
                  labels=[f'{round(x - 1 / num_bins, 2)}-{round(x, 2)}' for x in np.arange(1 / num_bins, 1.0001, 1 / num_bins)])


folder_names = ['mlp_dp_sgd', 'mlp_dp_loss', 'lstm_dp_sgd', 'lstm_dp_loss']
results_dict = {folder: [] for folder in folder_names}
num_bins = 4  # Number of bins to split scores into
k_for_knn = 4  # Number of nearest neighbours to use

"""Manage original data"""
week_num = 3
LA_num = 11
df_original = pd.read_csv('transform_dataset_to_time_series_student_activity_withLA_withScore_v8.csv')
df_original = df_original.drop(columns=['student_id', 'program_id', 'team_id', 'start_date'])

x_original = df_original[df_original.columns[0:11 * 21]].values

# create a dataframe with weekly activity counts
reshape_syn_time_series = np.reshape(x_original, (len(x_original) * week_num * LA_num, 7))
agg_reshape_syn_time_series = np.sum(reshape_syn_time_series, axis=1)
x_original = np.reshape(agg_reshape_syn_time_series, (len(x_original), week_num * LA_num))

binned_week1_original = split_df_into_bins(df_original, 'score_week1', num_bins)
binned_week2_original = split_df_into_bins(df_original, 'score_week2', num_bins)
binned_week3_original = split_df_into_bins(df_original, 'score_week3', num_bins)

y_original = pd.concat([binned_week1_original, binned_week2_original, binned_week3_original], axis=1).values

train_x_original, test_x_original, train_y_original, test_y_original = \
    train_test_split(x_original, y_original, test_size=0.2, random_state=999)

"""Original -> original"""
knn_original = KNeighborsClassifier(n_neighbors=k_for_knn, p=2)
knn_original.fit(train_x_original, train_y_original)

predictions_original_to_original = knn_original.predict(test_x_original)
accuracy_original_to_original = np.mean(predictions_original_to_original == np.array(test_y_original))
print('O->O Accuracy of the test dataset: ', accuracy_original_to_original)
print()

for folder_name in folder_names:
    run_folder_names = os.listdir(folder_name)
    y_values = []
    for run_folder_name in run_folder_names:
        csv_names = os.listdir(folder_name + '/' + run_folder_name)
        final_file = [fname for fname in csv_names if fname.endswith('teamids.csv')][0]
        filename = folder_name + '/' + run_folder_name + '/' + final_file

        run_dict = {}

        """Manage Synthetic Data"""
        # Drop some samples due to inconsistent lengths
        df_synthetic = pd.read_csv(filename).drop(columns=['team_id']).iloc[0:len(df_original)]
        x_synthetic = df_synthetic[df_synthetic.columns[0:11 * 21]].values

        # create a dataframe with weekly activity counts
        reshape_syn_time_series = np.reshape(x_synthetic, (len(x_synthetic) * week_num * LA_num, 7))
        agg_reshape_syn_time_series = np.sum(reshape_syn_time_series, axis=1)
        x_synthetic = np.reshape(agg_reshape_syn_time_series, (len(x_synthetic), week_num * LA_num))

        binned_week1_synthetic = split_df_into_bins(df_synthetic, 'week1', num_bins)
        binned_week2_synthetic = split_df_into_bins(df_synthetic, 'week2', num_bins)
        binned_week3_synthetic = split_df_into_bins(df_synthetic, 'week3', num_bins)

        y_synthetic = pd.concat([binned_week1_synthetic, binned_week2_synthetic, binned_week3_synthetic], axis=1).values

        train_x_synthetic, test_x_synthetic, train_y_synthetic, test_y_synthetic = \
            train_test_split(x_synthetic, y_synthetic, test_size=0.2, random_state=999)


        """Synthetic -> Synthetic"""
        knn_synthetic = KNeighborsClassifier(n_neighbors=k_for_knn, p=2)
        knn_synthetic.fit(train_x_synthetic, train_y_synthetic)

        predictions_synthetic_to_synthetic = knn_synthetic.predict(test_x_synthetic)
        accuracy_synthetic_to_synthetic = np.mean(predictions_synthetic_to_synthetic == np.array(test_y_synthetic))
        print('S->S Accuracy of the test dataset: ', accuracy_original_to_original)
        print()
        run_dict['S->S'] = accuracy_synthetic_to_synthetic


        """Original -> Synthetic"""
        predictions_original_to_synthetic = knn_original.predict(test_x_synthetic)
        accuracy_original_to_synthetic = np.mean(predictions_original_to_synthetic == np.array(test_y_synthetic))
        print('O->S Accuracy of the test dataset: ', accuracy_original_to_original)
        print()
        run_dict['O->S'] = accuracy_original_to_synthetic


        """Synthetic -> Original"""
        predictions_synthetic_to_original = knn_synthetic.predict(test_x_original)
        accuracy_synthetic_to_original = np.mean(predictions_synthetic_to_original == np.array(test_y_original))
        print('S->O Accuracy of the test dataset: ', accuracy_synthetic_to_original)
        print()
        run_dict['S->O'] = accuracy_synthetic_to_original

        results_dict[folder_name].append(run_dict)
        y_values.append(y_synthetic)

    all_weeks = np.concatenate(y_values)
    week1_histogram = np.unique(all_weeks[:, 0], return_counts=True)
    week2_histogram = np.unique(all_weeks[:, 1], return_counts=True)
    week3_histogram = np.unique(all_weeks[:, 2], return_counts=True)

    fig, axs = plt.subplots(nrows=3)
    axs[0].set_title(f'{folder_name} distribution of scores')

    axs[0].bar(week1_histogram[0], week1_histogram[1])
    axs[1].bar(week2_histogram[0], week2_histogram[1])
    axs[2].bar(week3_histogram[0], week3_histogram[1])
    plt.savefig(f'{folder_name}_knn_week_distribution.png')
    plt.close()


chart_dict = {}
for network_type, knn_dict in results_dict.items():
    results_csv = pd.DataFrame(knn_dict)
    results_csv.to_csv(network_type + '_knn_results.csv', index=False)
    chart_dict[network_type] = results_csv



width = 1 / (3 + 1)  # the width of the bars
utility_comparison_4cases_figure1 = plt.figure(figsize=(8, 6))
ax = utility_comparison_4cases_figure1.add_subplot(111)
error_bar_capsize = 6
alpha_value = 0.7

# Original Results
ax.bar(width, \
       accuracy_original_to_original, width * 3, \
       color='r', alpha=alpha_value)

for setting_idx, syn_data_prefix in enumerate(folder_names):
    sub_combined_utility_results = chart_dict[syn_data_prefix]

    ax.bar(setting_idx + 1 + width * 0, \
           np.mean(sub_combined_utility_results['O->S']), width, \
           yerr=np.std(sub_combined_utility_results['O->S']), \
           color='g', alpha=alpha_value, capsize=error_bar_capsize)
    ax.bar(setting_idx + 1 + width * 1, \
           np.mean(sub_combined_utility_results['S->S']), width, \
           yerr=np.std(sub_combined_utility_results['S->S']), \
           color='b', alpha=alpha_value, capsize=error_bar_capsize)
    ax.bar(setting_idx + 1 + width * 2, \
           np.mean(sub_combined_utility_results['S->O']), width, \
           yerr=np.std(sub_combined_utility_results['S->O']), \
           color='y', alpha=alpha_value, capsize=error_bar_capsize)

ax.set_ylabel('MSE')
ax.set_xticks(np.arange(len(folder_names) + 1) + width * 1)
ax.set_xticklabels(['Original'] + folder_names)


ax.legend(('Original model to original data (knn: test accuracy)', \
           'Original model to synthetic data (knn: pred. labels vs syn. labels)', \
           'Synthetic model to synthetic data (knn: test accuracy)', \
           'Synthetic model to original data (mse: knn. labels vs orig. labels)'), loc='upper left')
plt.savefig('knn_test.png')