import torch
import torch.utils.data
from torch import nn, optim, randn
import pandas as pd
import numpy as np
from torch.utils.data import random_split
import math
import matplotlib.pyplot as plt
import matplotlib
import random

import datetime
import os
from shutil import copy
from copy import deepcopy
from sklearn.preprocessing import StandardScaler


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
            nn.Linear(256, nx)
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
        return outputs


class DataLoaderDataset(torch.utils.data.Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index]


def get_noise_scale(N, num_epochs, batch_size, delta, input_epsilon):
    c_2 = 1.5
    q = batch_size / N
    T = num_epochs / q
    noise_scale = c_2 * q * np.sqrt(T * np.log(1 / delta)) / input_epsilon
    return noise_scale


def moving_average(x, w):
    convolved = pd.Series(x).rolling(window=w, min_periods=1).mean().values
    return convolved


def reverse_transform_noisy(sample, means, scales, inverse_transformation_sensitivity, N, inverse_eps):
    noisy_mean = means + \
                 np.random.laplace(np.zeros_like(means), (inverse_transformation_sensitivity / N / inverse_eps))
    noisy_scale = np.sqrt(np.abs(scales ** 2 +
                          np.random.laplace(np.zeros_like(scales), (inverse_transformation_sensitivity ** 2 / N / inverse_eps))))
    noisy_inverse = sample * noisy_scale + noisy_mean
    return noisy_inverse


"""Inputs"""
num_epochs = 500
nz = 10
batch_size = 16
N_disc = 1
plot_increment = 500
applying_scaling = True
rolling_mean_window = 4
clipping_parameter = 0.1

applying_noise = True

# 0) File and directory management
# Set up new directory for this execution. This will create a new folder with all the relevant saved content
# from this execution script. It will save a copy of this script, so you know which script was used to generate it
starting_time = '_'.join(str(datetime.datetime.now()).replace(':', '_').split())
# Try to make the new directory
path = f'occupancy_dp_gan_{starting_time}'
try:
    os.stat(path)
except:
    os.mkdir(path)
# Copy a timestamped version of this file into the new directory
copy(__file__, path + f'/occupancy_dp_gan_{starting_time}.py')

for iteration in range(1):
    torch.manual_seed(iteration)

    for epsilon in [9]:
        delta = 0.001
        N = 685
        discriminator_noise_scale = get_noise_scale(N, num_epochs, batch_size, delta, epsilon)
        min_empirical_discriminator_loss = 0
        max_empirical_discriminator_loss = 3

        inverse_transformation_sensitivity = 2100
        inverse_transform_eps = 1

        """End Inputs"""
        discriminator_noise_range = max_empirical_discriminator_loss - min_empirical_discriminator_loss


        # 1) Load data
        # Sanitise pandas inputs
        df = pd.read_csv('occupancy_binary.csv')
        occupancy = df['occupancy_level'].values
        df = df.drop(columns=['occupancy_level'])

        X_unscaled = df.values
        Scaling = StandardScaler()

        if applying_scaling:
            X = Scaling.fit_transform(X_unscaled)

            # Convert gaussian scaling object to be saved
            means_df = pd.DataFrame(Scaling.mean_)
            scales_df = pd.DataFrame(Scaling.scale_)
            means_df.to_csv(path + f'/means_df.csv', header=False, index=False)
            scales_df.to_csv(path + f'/scales_df.csv', header=False, index=False)
        else:
            X = X_unscaled


        # Manage imbalanced data
        counts = np.bincount(occupancy)
        labels_weights = 1. / counts
        weights = labels_weights[occupancy]
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights), replacement=True)

        allowed_samples_per_epoch = min(counts) * len(counts)

        # Load training data into dataloader (for compatibility with Pytorch)
        # dataset = SonarDataset(X_train, y_train)
        dataset = DataLoaderDataset(X)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            # shuffle=True,
            drop_last=True,
            sampler=sampler
        )


        nx = X.shape[1]
        netD = Discriminator(nx)
        netG = Generator(nz, nx)
        # netG = LSTMGenerator(nz, nx)

        # optimizerD = optim.SGD(netD.parameters(), lr=0.01,
        #                        momentum=0.9, weight_decay=1.0e-5)
        optimizerD = optim.Adam(netD.parameters(), lr=0.001, eps=1e-6,
                                betas=(0.5, 0.999), weight_decay=1e-5)
        optimizerG = optim.Adam(netG.parameters(), lr=0.001, eps=1e-6,
                                betas=(0.5, 0.999), weight_decay=1e-5)

        criterion = nn.BCEWithLogitsLoss()

        # Establish convention for real and fake labels during training
        one_vec = torch.ones(batch_size)  # Real
        zero_vec = torch.zeros(batch_size)  # Fake

        fixed_X = torch.Tensor(X)

        # Tracking variables
        G_losses = []
        D_losses = []
        all_errD = []
        all_errG = []
        all_min_of_all_l2_norms = []
        correlations = []
        max_correlations = []
        l2_norms = []
        min_L2_norms = []
        losses = []

        for epoch in range(num_epochs):
            current_epoch_count = 0
            for i, x in enumerate(dataloader):
                # Check if we have already viewed our privacy-allowed sample count for a given epoch
                current_epoch_count += len(x)
                if current_epoch_count > allowed_samples_per_epoch:
                    break

                random_z_vector = torch.randn(batch_size, nz)
                ############################
                # (1) Update D network
                ###########################
                # Train with all real batch
                # Zero gradients
                netD.zero_grad()
                # Pass the real samples to the discriminator
                errD_real = netD(x.float()).view(-1)
                # Calculate the loss based on error from the true value
                lossD_real = criterion(errD_real, one_vec)

                # if applying_noise:
                #     # Failure mode where loss becomes 0 for batch: skip this to avoid nan gradients
                #     if lossD_real.item() != 0:
                #         # Clamp loss to range based on experimental results for this network and dataset
                #         lossD_real_clamped = torch.clamp(lossD_real, min=min_empirical_discriminator_loss,
                #                                          max=max_empirical_discriminator_loss)
                #         # Apply noise to the loss
                #         lossD_real_clamped_noisy = lossD_real_clamped + \
                #                                    np.random.normal(0, discriminator_noise_scale * discriminator_noise_range) * \
                #                                    lossD_real_clamped / lossD_real_clamped.detach()
                #         # Backprop the discriminator score
                #         lossD_real_clamped_noisy.backward()
                # else:
                #     losses.append(lossD_real.item())
                #     lossD_real.backward()

                if applying_noise:
                    lossD_real.backward()

                    # # Clip norm of gradient vector to be maximum length C
                    nn.utils.clip_grad_norm_(netD.parameters(), clipping_parameter)

                    # Apply noise to each parameter gradient
                    for p in netD.parameters():
                        # Add noise
                        p.grad += torch.normal(torch.zeros_like(p), discriminator_noise_scale * clipping_parameter)

                # Train with all fake batch
                # Generate fake image batch using the generator
                # print(random_z_vector.shape)
                # v = netG(random_z_vector)
                # print(v.shape)
                errD_fake = netD(netG(random_z_vector).detach()).view(-1)
                # Calculate the gradients for this batch
                loss_D_fake = criterion(errD_fake, zero_vec)
                loss_D_fake.backward()

                # Step the discriminator optimizer
                optimizerD.step()

                # Get analytics for output (does not affect computation)
                D_x = errD_real.mean().item()
                D_G_z1 = errD_fake.mean().item()
                errD = errD_real - errD_fake
                all_errD.append(errD.mean().item())

                # Only train the generator once every N_disc iterations
                if i % N_disc == 0:
                    ############################
                    # (2) Update G network
                    ###########################
                    # Zero gradients
                    netG.zero_grad()
                    # Generate fake image batch using the generator
                    fake = netG(random_z_vector)  # MLP
                    # Score the fake image batch using the discriminator
                    errG = netD(fake).view(-1)
                    # Calculate gradients for G
                    loss_G = criterion(errG, one_vec)
                    loss_G.backward()
                    # Update G
                    optimizerG.step()

                    # Get analytics for output (does not affect computation)
                    D_G_z2 = errG.mean().item()
                    all_errG.append(errG.mean().item())

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
            print()

            # Plot every 100 epochs
            if (epoch % plot_increment == 0 and epoch != 0) or epoch == num_epochs - 1:
                with torch.no_grad():
                    # Set models to evaluation mode
                    netG.eval()

                    # Plot losses
                    plt.close()
                    plt.plot(range(len(G_losses)), G_losses, label='G_losses')
                    plt.plot(range(len(D_losses)), D_losses, label='D_losses')
                    plt.title('Generator and Discriminator Losses as models are trained')
                    plt.xlabel('Number of Epochs')
                    plt.ylabel('Loss value')
                    plt.legend(loc='upper right')
                    plt.savefig(path + f'/training.png')
                    plt.close()

                    # Generate num_fakes fake samples for analytics
                    num_fakes = 100
                    random_z_vector = torch.randn(num_fakes, nz)
                    fake_samples = netG(random_z_vector).numpy()
                    if applying_scaling:
                        fake_samples = np.apply_along_axis(
                            lambda m: reverse_transform_noisy(m,
                                                              Scaling.mean_,
                                                              Scaling.scale_,
                                                              inverse_transformation_sensitivity,
                                                              N,
                                                              inverse_transform_eps),
                            axis=1,
                            arr=fake_samples)

                    # Save a sample output
                    plt.plot(range(len(fake_samples[0].squeeze())), fake_samples[0].squeeze(), label='sample_output')
                    plt.title(f'Sample output at epoch {epoch}')
                    plt.xlabel('Time')
                    plt.ylabel('Output')
                    plt.legend(loc='upper right')
                    plt.savefig(path + f'/epoch_{epoch}.png')
                    plt.close()

                    # Save a few random outputs for comparison
                    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(5, 15))
                    plt.subplots_adjust(hspace=0.4)
                    ax1.plot(range(len(fake_samples[0].squeeze())), fake_samples[0].squeeze(),
                             label='sample_output_1', color='r')
                    ax1.plot(range(len(fake_samples[1].squeeze())), fake_samples[1].squeeze(),
                             label='sample_output_2', color='b')
                    ax1.plot(range(len(fake_samples[2].squeeze())), fake_samples[2].squeeze(),
                             label='sample_output_3', color='g')
                    ax1.set_title(f'3 Generated fake samples compared batch at epoch {epoch}')
                    ax1.set_ylabel('Output')
                    ax2.plot(range(len(fake_samples[0].squeeze())), fake_samples[0].squeeze(),
                             label='sample_output', color='r')
                    ax2.set_title(f'Sample output at epoch {epoch}')
                    ax2.set_ylabel('Output')
                    ax3.plot(range(len(fake_samples[1].squeeze())), fake_samples[1].squeeze(),
                             label='sample_output', color='b')
                    ax3.set_title(f'Sample output at epoch {epoch}')
                    ax3.set_ylabel('Output')
                    ax4.plot(range(len(fake_samples[2].squeeze())), fake_samples[2].squeeze(),
                             label='sample_output', color='g')
                    ax4.set_title(f'Sample output at epoch {epoch}')
                    ax4.set_ylabel('Output')
                    plt.savefig(path + f'/epoch_{epoch}_batch.png')
                    plt.close()

                    # # Save fake outputs for comparison
                    # f, axes_array = plt.subplots(num_fakes, 1, figsize=(15, 5 * num_fakes))
                    # plt.subplots_adjust(hspace=0.4)
                    # for fake_sample, axis in zip(fake_samples, axes_array):
                    #     axis.plot(range(len(fake_sample.squeeze())), fake_sample.squeeze(),
                    #               label='fake', color='b')
                    #     closest_original_index = np.argmin(
                    #         list(np.linalg.norm(fake_sample.squeeze() - original) for original in X_unscaled))
                    #     axis.plot(range(len(X_unscaled[closest_original_index])), X_unscaled[closest_original_index],
                    #               label='closest original', color='orange')
                    #     axis.legend()
                    # plt.savefig(path + f'/epoch_{epoch}_fake_samples.png')
                    # plt.close()
                    #
                    # # Save 100 real outputs with closest fake sample comparison
                    # num_original_samples = 100
                    # random_X_index = np.random.choice(len(X), size=num_original_samples, replace=False)
                    # random_X = X_unscaled[random_X_index, :]
                    # f, axes_array = plt.subplots(len(random_X), 1, figsize=(
                    # 15, 5 * num_original_samples))  # Hardcoded size but limitation of max dimension
                    # plt.subplots_adjust(hspace=0.4)
                    # for original, axis in zip(random_X, axes_array):
                    #     axis.plot(range(len(original)), original, label='original', color='orange')
                    #     closest_fake_index = np.argmin(
                    #         list(np.linalg.norm(fake_sample.squeeze() - original) for fake_sample in fake_samples))
                    #     axis.plot(range(len(fake_samples[closest_fake_index])), fake_samples[closest_fake_index],
                    #               label='closest fake', color='b')
                    #     axis.legend()
                    # plt.savefig(path + f'/epoch_{epoch}_closest_fake_to_real_samples.png')
                    # plt.close()

                    # # Check correlation and L2 norms with original samples
                    # all_correlations = []
                    # all_l2_norms = []
                    # min_of_all_l2_norms = []
                    # for fake_sample in fake_samples:
                    #     min_of_all_l2_norms.append(min(np.linalg.norm(fake_sample.squeeze() - x) for x in X_unscaled))
                    #     for x in X_unscaled:
                    #         # Store the samples for a given correlation and L2 norm
                    #         all_l2_norms.append({
                    #             'norm': np.linalg.norm(fake_sample.squeeze() - x),
                    #             'samples': (fake_sample.squeeze(), x)})
                    #
                    #         all_correlations.append({
                    #             'correlation': np.correlate(fake_sample.squeeze(), x),
                    #             'samples': (fake_sample.squeeze(), x)})
                    #
                    # all_min_of_all_l2_norms.append(sum(min_of_all_l2_norms) / len(min_of_all_l2_norms))
                    #
                    # correlations.append(sum(pairing['correlation'] for pairing in all_correlations) / len(all_correlations))
                    # max_correlation = max(all_correlations, key=lambda p: p['correlation'])
                    # max_correlations.append(max_correlation['correlation'])
                    #
                    # l2_norms.append(sum(pairing['norm'] for pairing in all_l2_norms) / len(all_l2_norms))
                    # min_l2_norm = min(all_l2_norms, key=lambda p: p['norm'])
                    # min_L2_norms.append(min_l2_norm['norm'])
                    # # Save correlation output
                    # # Plot correlations
                    # x_vals_for_plot = [x * plot_increment for x in range(0, len(correlations))]
                    # plt.plot(x_vals_for_plot, correlations, label='correlation')
                    # plt.title(
                    #     f'AVERAGE Correlation between {num_fakes} generated samples \nand'
                    #     f'original data as models are trained')
                    # plt.xlabel('num epochs')
                    # plt.ylabel('Correlation')
                    # plt.legend(loc='upper right')
                    # plt.savefig(path + f'/correlation.png')
                    # plt.close()
                    # # Plot max correlations
                    # plt.plot(x_vals_for_plot, max_correlations, label='max correlation')
                    # plt.title(f'MAX Correlation between {num_fakes} generated samples \nand '
                    #           f'original data as models are trained')
                    # plt.xlabel('num epochs')
                    # plt.ylabel('Max Correlation')
                    # plt.legend(loc='upper right')
                    # plt.savefig(path + f'/max_correlation.png')
                    # plt.close()
                    #
                    # # Save L2 norm plot
                    # # Plot Average L2 norm
                    # plt.plot(x_vals_for_plot, l2_norms, label='l2_norms')
                    # plt.title(
                    #     f'AVERAGE L2 norm between {num_fakes} generated samples \nand '
                    #     f'original data as models are trained')
                    # plt.xlabel('num epochs')
                    # plt.ylabel('L2 Norm')
                    # plt.legend(loc='upper right')
                    # plt.savefig(path + f'/l2_norm_graph.png')
                    # plt.close()
                    # # Plot Min L2 Norm total
                    # plt.plot(x_vals_for_plot, min_L2_norms, label='total_min_l2_norms')
                    # plt.title(
                    #     f'MINIMUM Sum of L2 norm between {num_fakes} generated samples \nand '
                    #     f'original data as models are trained')
                    # plt.xlabel('num epochs')
                    # plt.ylabel('L2 Norm')
                    # plt.legend(loc='upper right')
                    # plt.savefig(path + f'/min_l2_norm_graph.png')
                    # plt.close()
                    #
                    # # Plot Min L2 Norm for min in batch
                    # plt.plot(x_vals_for_plot, all_min_of_all_l2_norms, label='min_l2_norms')
                    # plt.title(
                    #     f'MINIMUM L2 norm between {num_fakes} generated samples \nand '
                    #     f'original data as models are trained')
                    # plt.xlabel('num epochs')
                    # plt.ylabel('L2 Norm')
                    # plt.legend(loc='upper right')
                    # plt.savefig(path + f'/min_l2_norm_graph.png')
                    # plt.close()
                    #
                    # # Save the sample with the smallest L2 norm
                    # plt.plot(range(len(min_l2_norm['samples'][0])), min_l2_norm['samples'][0], label='fake')
                    # plt.plot(range(len(min_l2_norm['samples'][1])), min_l2_norm['samples'][1], label='real')
                    # plt.title(f'Smallest L2 norm at epoch {epoch}')
                    # plt.xlabel('Time')
                    # plt.ylabel('Output')
                    # plt.legend(loc='upper right')
                    # plt.savefig(path + f'/epoch_{epoch}_min_l2_norm.png')
                    # plt.close()
                    # # Save the sample with the max correlation
                    # plt.plot(range(len(max_correlation['samples'][0])), max_correlation['samples'][0], label='fake')
                    # plt.plot(range(len(max_correlation['samples'][1])), max_correlation['samples'][1], label='real')
                    # plt.title(f'Max correlation at epoch {epoch}')
                    # plt.xlabel('Time')
                    # plt.ylabel('Output')
                    # plt.legend(loc='upper right')
                    # plt.savefig(path + f'/epoch_{epoch}_max_correlation.png')
                    # plt.close()

                    # Save outputs
                    num_saved = len(df)
                    random_z_vector = torch.randn(num_saved, nz)
                    fake_samples_saving = netG(random_z_vector).numpy()
                    pd.DataFrame(fake_samples_saving).to_csv(path + f'/generated_fakes_normalised_{iteration}.csv', header=df.columns,
                                                             index=False)
                    if applying_scaling:
                        reverse_transformed_fakes = np.apply_along_axis(
                            lambda m: reverse_transform_noisy(m,
                                                              Scaling.mean_,
                                                              Scaling.scale_,
                                                              inverse_transformation_sensitivity,
                                                              N,
                                                              inverse_transform_eps),
                            axis=1,
                            arr=fake_samples_saving)
                        # Apply rolling mean to each fake sample
                        # Split into each category, to avoid producing the rolling mean over the category edge
                        reverse_transformed_fakes = np.concatenate([
                            np.apply_along_axis(
                                lambda m: moving_average(m, rolling_mean_window),
                                axis=1,
                                arr=reverse_transformed_fakes[:, 0:30]),
                            np.apply_along_axis(
                                lambda m: moving_average(m, rolling_mean_window),
                                axis=1,
                                arr=reverse_transformed_fakes[:, 30:60]),
                            np.apply_along_axis(
                                lambda m: moving_average(m, rolling_mean_window),
                                axis=1,
                                arr=reverse_transformed_fakes[:, 60:90])],
                            axis=1
                        )

                    pd.DataFrame(reverse_transformed_fakes).to_csv(path + f'/generated_fakes_{iteration}.csv', header=df.columns,
                                                                       index=False)

                    # Save models
                    torch.save(netG.state_dict(), path + '/netG.pth')
                    torch.save(netD.state_dict(), path + '/netD.pth')

                    # Set models back to training mode
                    netG.train()
