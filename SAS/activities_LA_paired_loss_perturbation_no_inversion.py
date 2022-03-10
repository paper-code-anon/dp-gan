import torch
import torch.utils.data
from torch import nn, optim, randn
import pandas as pd
import numpy as np
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
import matplotlib
import random

import datetime
import os
from shutil import copy
from copy import deepcopy

from sklearn.preprocessing import StandardScaler

# Use Ming's utility function to plot covariance matrix
# from time_series_utility_function_import_paired_gans import plot_cov_matrix_and_autocorr


class Generator(nn.Module):
    def __init__(self, nz, nx):
        super(Generator, self).__init__()
        # Generator maps nz -> nx
        self.main = nn.Sequential(
            nn.Linear(nz, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, nx),
        )

    def forward(self, input):
        return self.main(input)


class LSTMGenerator(nn.Module):
    """An LSTM based generator. It expects a sequence of noise vectors as input.
    Args:
        in_dim: Input noise dimensionality
        out_dim: Output dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms
    Input: noise of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, out_dim)
    """

    def __init__(self, in_dim, out_dim, n_layers=1, hidden_dim=32):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, out_dim)
    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        recurrent_features, _ = self.lstm(input.unsqueeze(1), (h_0, c_0))
        outputs = self.linear(recurrent_features.squeeze())
        # return recurrent_features
        # outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_dim))
        # outputs = outputs.view(batch_size, seq_len, self.out_dim)
        return outputs


class Discriminator(nn.Module):
    def __init__(self, nx):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nx, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, input):
        return self.main(input)


class GANDataset(torch.utils.data.Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index]


def reverse_transform_noisy_and_apply_relu(sample, Scaling_object, inverse_transformation_sensitivity, N, inverse_eps):
    noisy_mean = Scaling_object.mean_ + \
                 np.random.laplace(np.zeros_like(Scaling_object.mean_), (inverse_transformation_sensitivity / N / inverse_eps))
    noisy_scale = np.sqrt(np.abs(Scaling_object.scale_ ** 2 +
                          np.random.laplace(np.zeros_like(Scaling_object.scale_), (inverse_transformation_sensitivity ** 2 / N / inverse_eps))))
    noisy_inverse = sample * noisy_scale + noisy_mean
    noisy_inverse[noisy_inverse < 0] = 0
    return noisy_inverse


# class DimensionScaler:
#     def __init__(self):
#         self.mean = None
#         self.variance = None
#
#     def fit_transform(self, X_untransformed_single_dim):
#         # X_untransformed_single_dim has shape 1041, 21
#         self.mean = np.mean(X_untransformed_single_dim)
#         self.variance = np.var(X_untransformed_single_dim)
#         return (X_untransformed_single_dim - self.mean) / self.variance
#
#     def inverse_transform(self, X_transformed_single_dim):
#         if isinstance(X_transformed_single_dim, pd.DataFrame):
#             return (X_transformed_single_dim * self.variance + self.mean).values
#         else:
#             return X_transformed_single_dim * self.variance + self.mean


def run_gan(df, batch_size, num_epochs, discriminator_noise_scale):
    # -1) Input variables
    torch.manual_seed(99)  # Set manual random seed for torch random variables
    data_input = 'activities_LA_paired'
    matplotlib.use('agg')  # Turn off display of plots to save memory
    time_series_length = 21
    time_series_feature_length = 11
    nz = 10  # Dimension of latent variable space for our networks
    N_disc = 5  # Ratio of iterations to train the discriminator / training the generator
    plot_increment = 100  # How many epochs do we plot results
    clipping_parameter = 0.1

    # Discriminator noise perturbation values
    min_empirical_discriminator_loss = 0
    max_empirical_discriminator_loss = 8
    discriminator_noise_range = max_empirical_discriminator_loss - min_empirical_discriminator_loss

    # Inverse transformation configuration parameters
    inverse_transformation_sensitivity = 200

    # 0) File and directory management
    # Set up new directory for this execution. This will create a new folder with all the relevant saved content
    # from this execution script. It will save a copy of this script, so you know which script was used to generate it
    starting_time = '_'.join(str(datetime.datetime.now()).replace(':', '_').split())
    # Try to make the new directory
    path = f'{data_input}_gan_{starting_time}'
    try:
        os.stat(path)
    except:
        os.mkdir(path)
    # Copy a timestamped version of this file into the new directory
    copy(__file__, path + f'/{data_input}_gan_{starting_time}.py')
    # Copy any dependencies into the new directory
    # copy('time_series_utility_function_import.py', path + f'/time_series_utility_function_import_{starting_time}.py')

    # 1) Load data
    # df = pd.read_csv('transform_dataset_to_time_series_student_activity_withLA_withScore_v5.csv')
    df = df.drop(columns=['student_id', 'program_id', 'team_id', 'start_date', 'score_week1', 'score_week2', 'score_week3'])
    # Get learning activity order
    original_column_names = df.columns
    la_sequence_names = [activity.strip('_Day_0') for i, activity in enumerate(df.columns) if i % 21 == 0]
    X_unscaled_unshaped = np.reshape(df.values, (len(df),  time_series_feature_length, time_series_length))
    X_unscaled = np.swapaxes(X_unscaled_unshaped, 1, 2)

    # Test that input transformation is operating as expected
    X_test = np.zeros_like(X_unscaled)
    for i in range(len(df)):
        for j in range(time_series_length):
            X_test[i, j, :] = np.array([df.values[i, j * (time_series_feature_length): (j + 1) * time_series_feature_length]])
    assert sum(sum(sum(X_test - X_unscaled))) == 0

    Scalings = [StandardScaler() for _ in range(time_series_feature_length)]
    # Scalings = [DimensionScaler() for _ in range(time_series_feature_length)]
    X = np.zeros_like(X_unscaled)
    for i in range(time_series_feature_length):
        X[:, :, i] = Scalings[i].fit_transform(X_unscaled[:, :, i])

    dataloader = torch.utils.data.DataLoader(
        dataset=GANDataset(X),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # 2) Generate Neural Network objects with parameters
    # Define a custom weights initialization function to start with small normally distributed weights
    # This seems to help to keep the gradients from blowing up if there is no clipping
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 2)  # TODO: Find source for these
            m.bias.data.fill_(0)

    # Create 11 generator discriminator pairs: 1 for each learning analytic
    nx = time_series_length
    netG_array = [Generator(nz, nx) for _ in range(time_series_feature_length)]
    # netG_array = [LSTMGenerator(nz, nx) for _ in range(time_series_feature_length)]

    netD_array = [Discriminator(nx).apply(weights_init) for _ in range(time_series_feature_length)]
    criterion = nn.BCEWithLogitsLoss()
    single_sample_criterion = nn.BCEWithLogitsLoss(reduce=False)

    # 3) Set up optimizers
    optimizerD_array = [optim.Adam(netD.parameters(), lr=0.001, eps=1e-6,
                            betas=(0.5, 0.999), weight_decay=1e-5) for netD in netD_array]
    optimizerG_array = [optim.Adam(netG.parameters(), lr=0.001, eps=1e-6,
                            betas=(0.5, 0.999), weight_decay=1e-5) for netG in netG_array]

    # 5) Setup necessary variables for iteration loop
    G_losses = []
    D_losses = []

    # Establish convention for real and fake labels during training
    one_vec = torch.ones(batch_size)  # Real
    zero_vec = torch.zeros(batch_size)  # Fake

    # 7) Start iterating through data
    for epoch in range(num_epochs):
        all_errG = []
        all_errD = []
        for i, x in enumerate(dataloader):
            # Generate a single sample of noise to be fed to each generator / discriminator pair
            random_z_vector = torch.randn(batch_size, nz)
            # Iterate through each generator / discriminator pair and train
            for sample_idx, netD_sample, netG_sample, optimizerD_sample, optimizerG_sample in \
                    zip(range(11), netD_array, netG_array, optimizerD_array, optimizerG_array):
                # Select only the x for the dimension we are training
                x_sample = x[:, :, sample_idx].float()
                ############################
                # (1) Update D network
                ###########################
                # Train with all real batch
                # Zero gradients
                netD_sample.zero_grad()
                # Pass the real samples to the discriminator
                errD_real = netD_sample(x_sample).view(-1)
                # Calculate the loss based on error from the true value
                lossD_real = single_sample_criterion(errD_real, one_vec)
                # lossD_real = criterion(errD_real, one_vec)

                # Clamp loss to range based on experimental results for this network and dataset
                # Add small value to avoid backprop on 0
                lossD_real_clamped = torch.clamp(lossD_real, min=min_empirical_discriminator_loss + 0.00001,
                                                 max=max_empirical_discriminator_loss)

                lossD_real_mean = lossD_real_clamped.mean()

                # Apply noise to the loss
                lossD_real_clamped_noisy = lossD_real_mean + \
                                           np.random.normal(0, discriminator_noise_scale * discriminator_noise_range) * \
                                           lossD_real_mean / lossD_real_mean.detach() / np.sqrt(batch_size)

                lossD_real_clamped_noisy.backward()
                    
                # applying_noise = True
                # if applying_noise:
                #     lossD_real.backward()
                
                #     # # Clip norm of gradient vector to be maximum length C
                #     nn.utils.clip_grad_norm_(netD_sample.parameters(), clipping_parameter)
                
                #     # Apply noise to each parameter gradient
                #     for p in netD_sample.parameters():
                #         # Add noise
                #         p.grad += torch.normal(torch.zeros_like(p), discriminator_noise_scale * clipping_parameter)

                # Train with all fake batch
                # Generate fake image batch using the generator
                errD_fake = netD_sample(netG_sample(random_z_vector).detach()).view(-1)
                # Calculate the gradients for this batch
                loss_D_fake = criterion(errD_fake, zero_vec)
                loss_D_fake.backward()

                # Step the discriminator optimizer
                optimizerD_sample.step()

                # Get analytics for output (does not affect computation)
                D_x = errD_real.mean().item()
                D_G_z1 = errD_fake.mean().item()
                errD = errD_real - errD_fake
                all_errD.append(errD.mean())

                # Only train the generator once every N_disc iterations
                if i % N_disc == 0:
                    ############################
                    # (2) Update G network
                    ###########################
                    # Zero gradients
                    netG_sample.zero_grad()
                    # Generate fake image batch using the generator
                    fake = netG_sample(random_z_vector)  # MLP
                    # Score the fake image batch using the discriminator
                    errG = netD_sample(fake).view(-1)
                    # Calculate gradients for G
                    loss_G = criterion(errG, one_vec)
                    loss_G.backward()
                    # Update G
                    optimizerG_sample.step()

                    # Get analytics for output (does not affect computation)
                    D_G_z2 = errG.mean().item()
                    all_errG.append(errG.mean())

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tErr_D: %.4f\tErr_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.mean(), errG.mean(), D_x, D_G_z1, D_G_z2))


        ###### Everything below here is just visualisation ######
        # Save Losses for plotting later
        G_losses.append(float(sum(all_errG) / len(all_errG)))
        D_losses.append(float(sum(all_errD) / len(all_errD)))

        print(f'[{epoch}/{num_epochs}] '
              f'D_x: {D_x}, '
              f'D_G_z1: {D_G_z1}, '
              f'D_G_z2: {D_G_z2}, ')

        with torch.no_grad():
            # Only save plots every plot_increment epochs to save time
            if epoch % plot_increment == 0 or epoch == num_epochs - 1:
                saved_vals_list = []
                # Iterate through all dimensions
                for sample_idx, netD_sample, netG_sample, optimizerD_sample, optimizerG_sample, Gaussian_scaling in \
                        zip(range(11), netD_array, netG_array, optimizerD_array, optimizerG_array, Scalings):
                    # Set models to evaluation mode
                    netG_sample.eval()

                    # Plot losses
                    # plt.close()
                    # plt.plot(range(len(G_losses)), G_losses, label='G_losses')
                    # plt.plot(range(len(D_losses)), D_losses, label='D_losses')
                    # plt.title('Generator and Discriminator Losses as models are trained')
                    # plt.xlabel('Number of Epochs')
                    # plt.ylabel('Loss value')
                    # plt.legend(loc='upper right')
                    # plt.savefig(path + f'/training.png')
                    # plt.close()

                    # Generate num_fakes fake samples for analytics
                    num_fakes = 100
                    random_z_vector = torch.randn(num_fakes, nz)
                    fake_samples = netG_sample(random_z_vector).numpy()

                    # Save a sample output
                    # plt.plot(range(len(fake_samples[0].squeeze())),
                    #          reverse_transform_noisy_and_apply_relu(fake_samples[0].squeeze(),
                    #                                                 Gaussian_scaling,
                    #                                                 inverse_transformation_sensitivity,
                    #                                                 len(df),
                    #                                                 transformation_epsilon),
                    #          label='sample_output')
                    # plt.title(f'Sample output at epoch {epoch}')
                    # plt.xlabel('Time')
                    # plt.ylabel('Output')
                    # plt.legend(loc='upper right')
                    # plt.savefig(path + f'/epoch_{epoch}_dim_{sample_idx}.png')
                    # plt.close()
                    #
                    # # Save a few random outputs for comparison
                    # f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(5, 15))
                    # plt.subplots_adjust(hspace=0.4)
                    # ax1.plot(range(len(fake_samples[0].squeeze())),
                    #          reverse_transform_noisy_and_apply_relu(fake_samples[0].squeeze(),
                    #                                                 Gaussian_scaling,
                    #                                                 inverse_transformation_sensitivity,
                    #                                                 len(df),
                    #                                                 transformation_epsilon),
                    #          label='sample_output_1', color='r')
                    # ax1.plot(range(len(fake_samples[1].squeeze())),
                    #          reverse_transform_noisy_and_apply_relu(fake_samples[1].squeeze(),
                    #                                                 Gaussian_scaling,
                    #                                                 inverse_transformation_sensitivity,
                    #                                                 len(df),
                    #                                                 transformation_epsilon),
                    #          label='sample_output_2', color='b')
                    # ax1.plot(range(len(fake_samples[2].squeeze())),
                    #          reverse_transform_noisy_and_apply_relu(fake_samples[2].squeeze(),
                    #                                                 Gaussian_scaling,
                    #                                                 inverse_transformation_sensitivity,
                    #                                                 len(df),
                    #                                                 transformation_epsilon),
                    #          label='sample_output_3', color='g')
                    # ax1.set_title(f'3 Generated fake samples compared batch at epoch {epoch}')
                    # ax1.set_ylabel('Output')
                    # ax2.plot(range(len(fake_samples[0].squeeze())),
                    #          reverse_transform_noisy_and_apply_relu(fake_samples[0].squeeze(),
                    #                                                 Gaussian_scaling,
                    #                                                 inverse_transformation_sensitivity,
                    #                                                 len(df),
                    #                                                 transformation_epsilon),
                    #          label='sample_output', color='r')
                    # ax2.set_title(f'Sample output at epoch {epoch}')
                    # ax2.set_ylabel('Output')
                    # ax3.plot(range(len(fake_samples[1].squeeze())),
                    #          reverse_transform_noisy_and_apply_relu(fake_samples[1].squeeze(),
                    #                                                 Gaussian_scaling,
                    #                                                 inverse_transformation_sensitivity,
                    #                                                 len(df),
                    #                                                 transformation_epsilon),
                    #          label='sample_output', color='b')
                    # ax3.set_title(f'Sample output at epoch {epoch}')
                    # ax3.set_ylabel('Output')
                    # ax4.plot(range(len(fake_samples[2].squeeze())),
                    #          reverse_transform_noisy_and_apply_relu(fake_samples[2].squeeze(),
                    #                                                 Gaussian_scaling,
                    #                                                 inverse_transformation_sensitivity,
                    #                                                 len(df),
                    #                                                 transformation_epsilon),
                    #          label='sample_output', color='g')
                    # ax4.set_title(f'Sample output at epoch {epoch}')
                    # plt.xlabel('Day')
                    # ax4.set_ylabel('Output')
                    # plt.savefig(path + f'/epoch_{epoch}_dim_{sample_idx}_batch.png')
                    # plt.close()

                    # Calculate covariance matrix and autocorrelation with Ming's code
                    # plot_cov_matrix_and_autocorr(path, epoch, X_unscaled[:, :, sample_idx], fake_samples,
                    #                              Gaussian_scaling, sample_idx)

                    # Generate some number of vectors for saving as output
                    # Generate the same number of samples as the input
                    num_saved = len(df)
                    random_z_vector_saved = torch.randn(num_saved, nz)
                    fake_samples_saved = netG_sample(random_z_vector_saved).numpy()

                    fake_sample_array = []
                    for fake_sample in fake_samples_saved:
                        fake_sample_array.append(fake_sample)

                    single_dim_df = pd.DataFrame(fake_sample_array)
                    # # Apply noise for private inverse transformation
                    # single_dim_df *= np.random.laplace(1, (inverse_transformation_sensitivity ** 2 / num_saved) / transformation_epsilon)
                    # single_dim_df += np.random.laplace(0, (inverse_transformation_sensitivity / num_saved) / transformation_epsilon)
                    #
                    # # Floor outputs at 0
                    # single_dim_df = single_dim_df.applymap(lambda x: 0 if x < 0 else x)

                    saved_vals_list.append(single_dim_df)

                    # Checkpoint models
                    torch.save(netG_sample.state_dict(), path + f'/netG_dim_{sample_idx}.pth')
                    torch.save(netD_sample.state_dict(), path + f'/netD_dim_{sample_idx}.pth')

                    # Set models back to training mode
                    netG_sample.train()
                    netD_sample.train()

                # Concatenate all dimensions and save output
                output_dataframe = pd.concat(saved_vals_list, axis=1)
                output_dataframe.columns = original_column_names
                output_dataframe.to_csv(path + '/gan_generated_time_series.csv', header=True, index=False)

    return output_dataframe, Scalings

