import pandas as pd
import numpy as np
import math

from scipy.io import loadmat

median_filter_window_size = 8
sequence_length = 100
subjects = range(1, 8)
iterations = range(1, 6)
activities = ['step', 'rest', 'squat']
column_names = ['accel_x'] + [f'accel_x.{i}' for i in range(1, 100)] +\
               ['accel_y'] + [f'accel_y.{i}' for i in range(1, 100)] +\
               ['accel_z'] + [f'accel_z.{i}' for i in range(1, 100)] +\
               ['ppg'] + [f'ppg.{i}' for i in range(1, 100)] +\
               ['activity', 'subject']


def manage_ppgs(activity, subject, iteration):
    folder_name = f'S{subject}'
    filename = f'{activity}{iteration}_ppg.mat'
    f = loadmat(f'PPG_ACC_dataset/{folder_name}/{filename}')
    ppgs = f['PPG'][:, 1]
    # print(ppgs.shape)

    max_chunk = math.floor(len(ppgs) / 800) * 800
    ppgs = ppgs[0:max_chunk]

    ppg_with_median = np.concatenate([np.median(ppgs[i * median_filter_window_size: (i + 1) * median_filter_window_size])
         for i in range(math.floor(len(ppgs) / median_filter_window_size))], axis=None)

    number_of_sequences = math.floor(len(ppg_with_median) / sequence_length)
    split_ppg_with_median = np.split(ppg_with_median[0:number_of_sequences * sequence_length], number_of_sequences)
    stacked_split_ppg_with_median = np.stack(split_ppg_with_median, axis=0)

    return stacked_split_ppg_with_median

def manage_xyz(activity, subject, iteration):
    # activity = 'rest'
    # subject = 1
    # iteration = 1
    folder_name = f'S{subject}'
    filename = f'{activity}{iteration}_acc.mat'
    f = loadmat(f'PPG_ACC_dataset/{folder_name}/{filename}')
    xyz = f['ACC'][:, 1:4]
    # print(xyz.shape)
    parsed = []
    for i in range(3):
        v = xyz[:, i]
        max_chunk = math.floor(len(v) / 800) * 800
        # print(max_chunk, len(v))
        v = v[0:max_chunk]
        # print(v[0:8])
        v_with_median = np.concatenate(
            [np.median(v[i * median_filter_window_size: (i + 1) * median_filter_window_size])
             for i in range(math.floor(len(v) / median_filter_window_size))], axis=None)
        number_of_sequences = math.floor(len(v_with_median) / sequence_length)
        split_v_with_median = np.split(v_with_median[0:number_of_sequences * sequence_length], number_of_sequences)
        stacked_split_v_with_median = np.stack(split_v_with_median, axis=0)
        parsed.append(stacked_split_v_with_median)

    concat_xyz = np.concatenate(parsed, axis=1)
    # print(concat_xyz)
    # quit()
    return concat_xyz


rows = []
for subject_index, subject in enumerate(subjects):
    for iteration in iterations:
        for activity_index, activity in enumerate(activities):
            print(subject, activity, iteration)
            xyzs = manage_xyz(activity, subject, iteration)
            ppgs = manage_ppgs(activity, subject, iteration)
            activity_label = (activity_index + 1) * np.ones((ppgs.shape[0], 1))
            subject_label = (subject_index + 1) * np.ones((ppgs.shape[0], 1))
            print(xyzs.shape)
            print(ppgs.shape)
            batch_rows = np.concatenate([xyzs, ppgs, activity_label, subject_label], axis=1)
            rows.append(batch_rows)

output = np.concatenate(rows, axis=0)
print(output.shape)
df = pd.DataFrame(output, columns=column_names)
# df.to_csv('test_ppg.csv', header=False, index=False)


# First fix any first sequences that are nans
for col in ['accel_x', 'accel_y', 'accel_z', 'ppg']:
    df[col] = df[col].fillna(df[col + '.1'])

# Now fix any other occurrences with the preceding value
for i in range(1, len(df.columns)):
    df[df.columns[i]] = df[df.columns[i]].fillna(df[df.columns[i - 1]])


activity_label = df['activity']#.apply(lambda x: activity_label_mapping[x]).values
# df = df.drop(columns=['activity', 'subject'])

# Manage imbalanced data
counts = np.bincount(activity_label)

activity_label_1 = df[df['activity'] == 1]
activity_label_2 = df[df['activity'] == 2]
activity_label_3 = df[df['activity'] == 3]

undersampled_2_index = np.random.choice(activity_label_2.shape[0], size=max(len(activity_label_1), len(activity_label_3)), replace=False)
undersampled_2 = activity_label_2.iloc[undersampled_2_index]

output_df = pd.concat([activity_label_1, undersampled_2, activity_label_3], axis=0)
output_df.to_csv('accel_x_y_z+ppgv3_removed_nulls_balanced.csv', index=False)


# df.to_csv('accel_x_y_z+ppgv3_removed_nulls_test.csv', index=False)
