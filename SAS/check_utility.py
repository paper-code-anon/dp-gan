# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:18:06 2020

@author: din031
"""


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def save_utility_image(folder_name, filename):
    syn_data_name = folder_name + '/' + filename

    analysis_sample_num = 100

    # enabler of clustering
    time_series_clustering_enabler = 0

    # number of LA categories
    LA_num = 11

    # number of weeks
    week_num = 3

    fake_data_reshape_enabler = 1

    # load the original time series
    orig_data_option = 'Practera_v3'

    # the Practera activity count dataset (LA specific) + weekly scores
    full_orig_data = pd.read_csv('transform_dataset_to_time_series_student_activity_withLA_withScore_v8.csv')

    # drop the rows with zero weekly scores
    drop_indices = full_orig_data[np.sum(full_orig_data[full_orig_data.columns[-3:]], axis=1)==0].index
    if len(drop_indices) > 0:
        full_orig_data = full_orig_data.drop(drop_indices)

    # drop the rows with zero daily activities
    drop_indices = full_orig_data[np.sum(full_orig_data[full_orig_data.columns[4:-3]], axis=1)==0].index
    if len(drop_indices) > 0:
        full_orig_data = full_orig_data.drop(drop_indices)

    # save the time series to temp_data
    temp_data = full_orig_data[full_orig_data.columns[4:-3]]

    # build a linear regression model mapping time series to scores with cross validation
    reshape_time_series = np.reshape(temp_data.values, (len(temp_data)*week_num*LA_num, 7))
    agg_reshape_time_series = np.sum(reshape_time_series, axis=1)
    weekly_time_series = np.reshape(agg_reshape_time_series, (len(temp_data), week_num*LA_num))


    # create a dataframe with weekly activity counts
    df_x = pd.DataFrame(weekly_time_series)
    y = full_orig_data[full_orig_data.columns[-3:]]

    from sklearn import datasets, linear_model
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score, cross_val_predict
    from sklearn import metrics

    # create training and testing vars
    X_train, X_test, y_train, y_test = \
        train_test_split(df_x, y['score_week3'], test_size=0.2, random_state=999)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # fit a linear regression model
    lm = linear_model.LinearRegression()
    model = lm.fit(X_train, y_train)
    predictions = lm.predict(X_test)


    # check the score of the fitting
    print('Model score: ', model.score(X_test, y_test))
    accuracy = metrics.r2_score(y_test, predictions)
    print('R-squared score: ', accuracy)
    MSE_test = np.mean((y_test-predictions)**2)
    print('MSE of the test dataset: ', MSE_test)




    # load the synthetic time series
    # be sure to remove the header before loading the csv file
    #full_syn_data = pd.read_csv(syn_data_name+'.csv', header=None)
    # TODO: Note this bug
    full_syn_data = pd.read_csv(syn_data_name).iloc[0:1000]
    if 'team_id' in full_syn_data.columns:
        full_syn_data = full_syn_data.drop(columns='team_id')

    syn_data = full_syn_data[full_syn_data.columns[0:LA_num*week_num*7]]
    full_syn_activity_counts = full_syn_data[full_syn_data.columns[0:LA_num*week_num*7]]
    syn_y = full_syn_data[['week1', 'week2', 'week3']]


    # create a dataframe with weekly activity counts
    reshape_syn_time_series = np.reshape(syn_data.values, (len(syn_data)*week_num*LA_num, 7))
    agg_reshape_syn_time_series = np.sum(reshape_syn_time_series, axis=1)
    weekly_syn_time_series = np.reshape(agg_reshape_syn_time_series, (len(syn_data), week_num*LA_num))

    # predict the scores from the synthetic activity data
    df_syn_x = pd.DataFrame(weekly_syn_time_series)
    predictions = lm.predict(df_syn_x)


    # check the score of the fitting
    print('Model score: ', model.score(df_syn_x, syn_y[syn_y.columns[2]]))
    accuracy = metrics.r2_score(syn_y[syn_y.columns[2]], predictions)
    print('R-squared score: ', accuracy)
    MSE_syn = np.mean((syn_y[syn_y.columns[2]]-predictions)**2)
    print('MSE of the synthetic dataset: ', MSE_syn)


    # visualize the linear regression prediction
    figure_lm_test = plt.figure(figsize=(8,8))
    plt.scatter(syn_y[syn_y.columns[2]], predictions, label='Synthetic', color='r')
    plt.title('Linear regression on the synthetic dataset [MSE = {0}, model score = {1}]'.format(np.round(MSE_test, 3), np.round(accuracy, 3)))
    plt.xlabel('Synthetic values of score_week3')
    plt.ylabel('Predicted values [LM built from the original data]')
    plt.legend(loc='best')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    # plt.show()

    plt.savefig(syn_data_name.strip('.csv') + '.png')

# if __name__ == '__main__':
#     check_utility('.', 'input_filename_here')