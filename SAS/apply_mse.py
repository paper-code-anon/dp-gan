
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
import os
import matplotlib.pyplot as plt


# filename = './mlp_dp_loss_compiled/mlp_dp_loss_0_eps_2_2_5_1/gan_time_series_eps_2_with_inversetransform_eps_2_with_scores_eps_5__with_threshold_eps_1_with_teamids.csv'
folder_names = ['mlp_dp_sgd', 'mlp_dp_loss', 'lstm_dp_sgd', 'lstm_dp_loss']
results_dict = {folder: [] for folder in folder_names}

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

y_original = pd.concat([df_original['score_week1'], df_original['score_week2'], df_original['score_week3']], axis=1).values
# y_original = df_original['score_week3'].values

train_x_original, test_x_original, train_y_original, test_y_original = \
    train_test_split(x_original, y_original, test_size=0.2, random_state=999)

"""Original -> original"""
lm_original = linear_model.LinearRegression()
model_original = lm_original.fit(train_x_original, train_y_original)

predictions_original_to_original = np.clip(lm_original.predict(test_x_original), 0, 1)
MSE_test_original = np.mean((test_y_original - predictions_original_to_original) ** 2)
print('O->O MSE of the test dataset: ', MSE_test_original)
print()

week1_histogram = np.histogram(y_original[:, 0], bins=20)
week2_histogram = np.histogram(y_original[:, 1], bins=20)
week3_histogram = np.histogram(y_original[:, 2], bins=20)

fig, axs = plt.subplots(nrows=3)
axs[0].set_title(f'Original distribution of scores')
axs[0].plot(week1_histogram[1][1:], week1_histogram[0])
axs[0].set_xlim(0, 1)
axs[1].plot(week2_histogram[1][1:], week2_histogram[0])
axs[1].set_xlim(0, 1)
axs[2].plot(week3_histogram[1][1:], week3_histogram[0])
axs[2].set_xlim(0, 1)
plt.savefig(f'original_week_distribution.png')
plt.close()


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

        y_synthetic = pd.concat([df_synthetic['week1'], df_synthetic['week2'], df_synthetic['week3']], axis=1).values
        # y_synthetic = df_synthetic['week3'].values
        train_x_synthetic, test_x_synthetic, train_y_synthetic, test_y_synthetic = \
            train_test_split(x_synthetic, y_synthetic, test_size=0.2, random_state=999)


        """Synthetic -> Synthetic"""
        lm_synthetic = linear_model.LinearRegression()
        model_synthetic = lm_synthetic.fit(train_x_synthetic, train_y_synthetic)
        predictions_synthetic_to_synthetic = np.clip(lm_synthetic.predict(test_x_synthetic), 0, 1)
        MSE_test = np.mean((test_y_synthetic - predictions_synthetic_to_synthetic) ** 2)
        print('S->S MSE of the test dataset: ', MSE_test)
        print()
        run_dict['S->S'] = MSE_test


        """Original -> Synthetic"""
        predictions_original_to_synthetic = np.clip(lm_original.predict(test_x_synthetic), 0, 1)
        MSE_test = np.mean((test_y_synthetic - predictions_original_to_synthetic) ** 2)
        print('O->S MSE of the test dataset: ', MSE_test)
        print()
        run_dict['O->S'] = MSE_test


        """Synthetic -> Original"""
        predictions_synthetic_to_original = np.clip(lm_synthetic.predict(test_x_original), 0, 1)
        MSE_test = np.mean((test_y_original - predictions_synthetic_to_original) ** 2)
        print('S->O MSE of the test dataset: ', MSE_test)
        run_dict['S->O'] = MSE_test

        results_dict[folder_name].append(run_dict)
        y_values.append(y_synthetic)

    all_weeks = np.concatenate(y_values)
    week1_histogram = np.histogram(all_weeks[:, 0], bins=20)
    week2_histogram = np.histogram(all_weeks[:, 1], bins=20)
    week3_histogram = np.histogram(all_weeks[:, 2], bins=20)

    fig, axs = plt.subplots(nrows=3)
    axs[0].set_title(f'{folder_name} distribution of scores')
    axs[0].plot(week1_histogram[1][1:], week1_histogram[0])
    axs[0].set_xlim(0, 1)
    axs[1].plot(week2_histogram[1][1:], week2_histogram[0])
    axs[1].set_xlim(0, 1)
    axs[2].plot(week3_histogram[1][1:], week3_histogram[0])
    axs[2].set_xlim(0, 1)
    plt.savefig(f'{folder_name}_week_distribution.png')
    plt.close()

chart_dict = {}
for network_type, mse_dict in results_dict.items():
    results_csv = pd.DataFrame(mse_dict)
    results_csv.to_csv(network_type + '_results.csv', index=False)
    chart_dict[network_type] = results_csv



width = 1 / (3 + 1)  # the width of the bars
utility_comparison_4cases_figure1 = plt.figure(figsize=(8, 6))
ax = utility_comparison_4cases_figure1.add_subplot(111)
error_bar_capsize = 6
alpha_value = 0.7

# Original Results
ax.bar(width, \
       MSE_test_original, width * 3, \
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


ax.legend(('Original model to original data (test mse)', \
           'Original model to synthetic data (mse: pred. labels vs syn. labels)', \
           'Synthetic model to synthetic data (test mse)', \
           'Synthetic model to original data (mse: pred. labels vs orig. labels)'), loc='upper left')
plt.savefig('mse_test.png')