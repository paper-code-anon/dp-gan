# transform the original dataset to a form that can be accepted by the time series synthesizer

import pandas as pd
from datetime import datetime
import numpy as np
#matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import os


# load all data
data1 = pd.read_csv('datatest.txt', header=0, index_col=1, parse_dates=True, squeeze=True)
data2 = pd.read_csv('datatraining.txt', header=0, index_col=1, parse_dates=True, squeeze=True)
data3 = pd.read_csv('datatest2.txt', header=0, index_col=1, parse_dates=True, squeeze=True)

# vertically stack and maintain temporal order
orig_data = pd.concat([data1, data2, data3])
data = orig_data.copy()
# delete rows with null values, if any
data = data.dropna(how='any', axis=0)

# drop the columns of date and humidity ratio
data = data.drop(columns=['date', 'HumidityRatio'])

# simplify the index
data.reset_index(drop=True, inplace=True)

# check the dataframe
data.columns
data.iloc[0]

# # save the combined dataset
# data.to_csv('combined_occupany_dataset.csv')

record_num = len(data)



# define the interested time series in the dataset
interested_time_series_name_list = ['Temperature', 'Humidity', 'Light', 'CO2']
interested_time_series_name_list = ['Temperature', 'Humidity', 'CO2']

multi_variate_time_series_name_list_array = np.array(interested_time_series_name_list, dtype=object)
multi_variate_time_series_num = len(multi_variate_time_series_name_list_array)

# generate the names of the time series columns
time_series_point_num = 30

time_series_columns_suffix = \
    np.array(np.tile(['TS_point_'], time_series_point_num), dtype=object) + \
    np.array(np.arange(time_series_point_num).astype(str), dtype=object)

# generate the time series column names with time series name prefix
time_series_columns = [x+'_'+time_series_columns_suffix for x in multi_variate_time_series_name_list_array]
time_series_columns = np.reshape(time_series_columns, \
                                (1, multi_variate_time_series_num*time_series_point_num))[0]

# get the attribute columns
trans_data_columns = ['time_granularity_unit_index']
trans_data_columns.extend(time_series_columns)

# add 1 column of occupancy level
target_attribute_column = ['occupancy_level']
trans_data_columns.extend(target_attribute_column)

# get the columns of time series and scores
time_and_occupancy_columns = list(time_series_columns)
time_and_occupancy_columns.extend(target_attribute_column)



# initialize the dataframe of the transformed data
trans_data_record_num = np.floor(record_num/time_series_point_num).astype(int)
trans_data = pd.DataFrame(0, \
                          index=np.arange(trans_data_record_num), \
                          columns=trans_data_columns)

# initialize the program_id to -999 so that we can remove abnormal records later
trans_data['time_granularity_unit_index'] = -999
trans_data.columns

# populate the transformed dataset
for trans_data_idx in range(trans_data_record_num):
    # select the half-hour time series from the original data
    sub_data = data.iloc[time_series_point_num*trans_data_idx : time_series_point_num*(trans_data_idx+1)]
    sub_time_series_matrix = sub_data[interested_time_series_name_list].values
    sub_time_series_matrix_to_array = np.reshape(sub_time_series_matrix.T, (1, time_series_point_num*multi_variate_time_series_num))
    
    # update the transformed dataset
    trans_data.loc[trans_data_idx, time_series_columns] = sub_time_series_matrix_to_array[0]
    trans_data.loc[trans_data_idx, 'time_granularity_unit_index'] = trans_data_idx
    trans_data.loc[trans_data_idx, 'occupancy_level'] = np.mean(sub_data['Occupancy'])







# visualize the target attribute
fig = plt.figure(figsize=(80,6))
plt.plot(trans_data['time_granularity_unit_index'], \
         trans_data[target_attribute_column[0]])
plt.xlabel('time_granularity_unit_index [time granularity = {0} minutes]'.format(time_series_point_num))
plt.ylabel(target_attribute_column[0])


# write the dataframe to a csv file to be process by RNN GAN
# Some students are skipped due to empty scores, drop these rows
trans_data = trans_data[trans_data['time_granularity_unit_index'] != -999]

occupancy_level = trans_data['occupancy_level']
trans_data = trans_data.drop(columns=['time_granularity_unit_index', 'occupancy_level'])

occupancy_level_binary = occupancy_level.apply(lambda x: 1 if x > 0 else 0)

output_df = pd.concat([trans_data, pd.DataFrame(occupancy_level_binary)], axis=1)
output_df.to_csv('occupancy_binary.csv', index=False)

# trans_data.to_csv('transform_room_sensor_time_series_with_occupancy_v1.csv', index=False)
#
# # test the output csv file
# test_data = pd.read_csv('transform_room_sensor_time_series_with_occupancy_v1.csv')
# test_data.iloc[0]
# test_data.columns
# len(test_data)
