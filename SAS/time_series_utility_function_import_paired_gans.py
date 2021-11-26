# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:18:06 2020

@author: din031
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns

from statsmodels.tsa import stattools
from sklearn.preprocessing import StandardScaler



def plot_cov_matrix_and_autocorr(path, epoch, orig_data_array, syn_data_array, Scaler, i):
    orig_data = pd.DataFrame(orig_data_array)
    syn_data = pd.DataFrame(syn_data_array)

    # get the time series length
    time_series_len = orig_data.shape[1]

    # get the sample numbers
    orig_sample_num = orig_data.shape[0]
    syn_sample_num = syn_data.shape[0]

    # # evaluate the normalized autocorrelation of the original time series data
    # # orig_autocorr_array = np.zeros(time_series_len*2-1)
    # orig_autocorr_array = np.zeros(time_series_len)
    # autocorr_x_axis_array = np.linspace(-time_series_len + 1, time_series_len - 1, len(orig_autocorr_array))
    # # plot the autocorrelation figure
    # figure_orig_time_series = plt.figure(figsize=(8, 6))
    # for sample_idx in range(orig_sample_num):
    #     #    orig_autocorr_array = orig_autocorr_array + \
    #     #        np.correlate(orig_data.iloc[sample_idx], orig_data.iloc[sample_idx], mode='full')
    #     orig_autocorr_array = orig_autocorr_array + \
    #                           stattools.acf(orig_data.iloc[sample_idx], \
    #                                             nlags=time_series_len, fft=False)
    #     if orig_sample_num <= 1e4:
    #         plt.plot(np.arange(time_series_len), orig_data.iloc[sample_idx], label='Sample #' + str(sample_idx))
    #
    # orig_autocorr_array = orig_autocorr_array / orig_sample_num
    #
    # if orig_sample_num <= 10:
    #     plt.legend(loc='best')

    # # evaluate the normalized autocorrelation of the synthetic time series data
    # syn_autocorr_array = np.zeros(time_series_len)
    # # autocorr_x_axis_array = np.linspace(-time_series_len+1, time_series_len-1, len(orig_autocorr_array))
    # # plot the autocorrelation figure
    # figure_syn_time_series = plt.figure(figsize=(8, 6))
    # for sample_idx in range(syn_sample_num):
    #     # if sum(syn_data.iloc[sample_idx]) > 0:
    #     syn_autocorr_array = syn_autocorr_array + \
    #                          stattools.acf(syn_data.iloc[sample_idx], \
    #                                            nlags=time_series_len, fft=False)
    #
    #     if syn_sample_num <= 1e4:
    #         plt.plot(np.arange(time_series_len), syn_data.iloc[sample_idx], label='Sample #' + str(sample_idx))
    #
    # syn_autocorr_array = syn_autocorr_array / syn_sample_num
    #
    # if syn_sample_num <= 10:
    #     plt.legend(loc='best')
    #
    # plt.title('Illustration of the synthetic samples (' + str(syn_sample_num) + ' samples)')
    # plt.xlabel('Time series point')
    # plt.ylabel('Sample value')
    # plt.savefig(path + f'/Illustration of the synthetic samples ({syn_sample_num} samples)_{epoch}_dim_{i}.png', format='png', dpi=200)
    # plt.close()

    if Scaler:
        # Illustration of true values
        # figure_true_values = plt.figure(figsize=(8, 6))
        # [plt.plot(orig_data.values[sample_idx]) for sample_idx in range(syn_sample_num)]
        # plt.title('Illustration of the true values samples (' + str(syn_sample_num) + ' samples)')
        # plt.xlabel('Time series point')
        # plt.ylabel('Sample value')
        # plt.savefig(path + f'/Illustration of the true values ({syn_sample_num} samples)_dim_{i}.png', format='png', dpi=200)
        # plt.close()

        # figure_reversed_transformed_time_series = plt.figure(figsize=(8, 6))
        inverse_transformed = Scaler.inverse_transform(syn_data)
        # [plt.plot(np.arange(time_series_len), inverse_transformed[sample_idx]) for sample_idx in range(syn_sample_num)]
        # plt.title('Illustration of the inverse transformed synthetic samples (' + str(syn_sample_num) + ' samples)')
        # plt.xlabel('Time series point')
        # plt.ylabel('Sample value')
        # plt.savefig(path + f'/Illustration of the inverse transformed synthetic samples ({syn_sample_num} samples)_{epoch}_dim_{i}.png', format='png', dpi=200)
        # plt.close()

        figure_reversed_transformed_relu_time_series = plt.figure(figsize=(8, 6))
        inverse_transformed[inverse_transformed < 0] = 0
        [plt.plot(np.arange(time_series_len), inverse_transformed[sample_idx]) for sample_idx in range(syn_sample_num)]
        plt.title('Illustration of the inverse transformed relu synthetic samples (' + str(syn_sample_num) + ' samples)')
        plt.xlabel('Time series point')
        plt.ylabel('Sample value')
        plt.savefig(path + f'/Illustration of the inverse transformed relu synthetic samples ({syn_sample_num} samples)_{epoch}_dim_{i}.png', format='png', dpi=200)
        plt.close()

        # generate the normalized covariance matrix
        syn_data_inverse_relu_norm_cov = pd.DataFrame(inverse_transformed).T.corr(method='pearson')

        # plot the correlation heatmap and annotation on it
        figure_syn_data_norm_cov = plt.figure(figsize=(20, 20))
        figure_handler = sns.heatmap(syn_data_inverse_relu_norm_cov, \
                                     cmap='jet', vmin=-1, vmax=1, \
                                     xticklabels=syn_data_inverse_relu_norm_cov.columns, \
                                     yticklabels=syn_data_inverse_relu_norm_cov.columns, annot=False)

        title_str = 'Normalized covariance matrix of the synthetic time series data with reverse transform relu'

        figure_handler.set_title(title_str, size=14)
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)
        plt.savefig(path + f'/figure_syn_data_norm_cov_reverse_transform_relu{epoch}_dim_{i}.png', format='png', dpi=200)
        plt.close()



        # orig_data_cov = pd.DataFrame(orig_data.values[0:100]).T.corr(method='pearson')
        # # plot the correlation heatmap of true values and annotation on it
        # plt.figure(figsize=(20, 20))
        # figure_handler = sns.heatmap(orig_data_cov, \
        #                              cmap='jet', vmin=-1, vmax=1, annot=False)
        #
        # title_str = 'Normalized covariance matrix of the true data'
        #
        # figure_handler.set_title(title_str, size=14)
        # plt.yticks(rotation=0)
        # plt.xticks(rotation=90)
        # plt.savefig(path + f'/true_dim_cov_matrix_dim_{i}.png', format='png', dpi=200)
        # plt.close()

    # figure_norm_autocorr_syn_time_series = plt.figure(figsize=(8, 6))
    # plt.plot(np.arange(time_series_len), syn_autocorr_array)
    # plt.title('Illustration of the synthetic autocorreation (normalized)')
    # plt.xlabel('Time lag')
    # plt.ylabel('Autocorrelation')
    # # plt.xlim([0, 600])
    # # plt.ylim([-1, 1])
    # plt.savefig(path + f'/Illustration of the autocorrelation (normalized)_{epoch}_dim_{i}.png', format='png', dpi=200)
    # plt.close()
    #
    # # plot the two autocorrelation figures in one figure
    # figure_norm_autocorr_comb = plt.figure(figsize=(8, 6))
    # plt.plot(np.arange(time_series_len), \
    #          orig_autocorr_array, 'b-', label='Original (' + str(orig_sample_num) + ' samples)')
    # plt.plot(np.arange(time_series_len), \
    #          syn_autocorr_array, 'r-', label='Synthetic (' + str(syn_sample_num) + ' samples)')
    # plt.title('Illustration of the autocorreation (normalized)')
    # plt.xlabel('Time lag')
    # plt.ylabel('Autocorrelation')
    # # plt.xlim([0, 600])
    # # plt.ylim([-1, 1])
    # plt.legend(loc='best')
    # plt.savefig(path + f'/Illustration of the autocorrelation (normalized)_compared_{epoch}_dim_{i}.png', format='png', dpi=200)
    # plt.close()

    # # generate the normalized covariance matrix
    # syn_data_norm_cov = syn_data.T.corr(method='pearson')
    #
    # # plot the correlation heatmap and annotation on it
    # figure_syn_data_norm_cov = plt.figure(figsize=(20, 20))
    # figure_handler = sns.heatmap(syn_data_norm_cov, \
    #                              cmap='jet', vmin=-1, vmax=1, \
    #                              xticklabels=syn_data_norm_cov.columns, \
    #                              yticklabels=syn_data_norm_cov.columns, annot=False)
    #
    # title_str = 'Normalized covariance matrix of the synthetic time series data'
    #
    # figure_handler.set_title(title_str, size=14)
    # plt.yticks(rotation=0)
    # plt.xticks(rotation=90)
    # plt.savefig(path + f'/figure_syn_data_norm_cov_{epoch}_dim_{i}.png', format='png', dpi=200)
    # plt.close()
