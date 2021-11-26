import torch
import torch.utils.data
from torch import nn, optim, randn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime
import os
from shutil import copy

from sklearn.model_selection import train_test_split


class Classifier(nn.Module):
    def __init__(self, nx):
        super(Classifier, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nx, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 3),
        )

    def forward(self, input):
        return self.main(input)


class ScoresDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def convert_la_to_scores(df_fakes, batch_size, num_epochs, noise_scale):
    # -1) Input parameters
    training_all = True  # Whether to merge the test and training set for producing final outpu

    # Noise application parameters
    applying_noise = True
    clipping_parameter = 0.2

    # # # 0) File and directory management
    # # Set up new directory for this execution. This will create a new folder with all the relevant saved content
    # # from this execution script. It will save a copy of this script, so you know which script was used to generate it
    # starting_time = '_'.join(str(datetime.datetime.now()).replace(':', '_').split())
    # # Try to make the new directory
    # path = f'score_classifier_{starting_time}'
    # try:
    #     os.stat(path)
    # except:
    #     os.mkdir(path)
    # # Copy a timestamped version of this file into the new directory
    # copy(__file__, path + f'/_{starting_time}.py')

    # 1) Load data
    time_series_dim = 21
    feature_dim = 11

    df = pd.read_csv('transform_dataset_to_time_series_student_activity_withLA_withScore_v8.csv').dropna()
    # Drop any cases where all scores are 0
    df = df[df['score_week1'] + df['score_week2'] + df['score_week3'] != 0]
    # Extract scores and activity values
    scores_raw = df[['score_week1', 'score_week2', 'score_week3']].values
    df = df.drop(columns=['student_id', 'program_id', 'team_id', 'start_date', 'score_week1', 'score_week2', 'score_week3'])
    X_unscaled = df.values

    # Split data into test and training set
    X_train, X_test, y_train, y_test = train_test_split(X_unscaled, scores_raw, test_size=0.2, train_size=0.8, random_state=32)
    if training_all:
        X_train = X_unscaled
        y_train = scores_raw

    train_dataloader = torch.utils.data.DataLoader(
        dataset=ScoresDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset=ScoresDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # Set up classifier and optimizer
    net = Classifier(time_series_dim * feature_dim)
    optimizer = optim.Adam(net.parameters(), lr=0.001, eps=1e-6, betas=(0.5, 0.999), weight_decay=1e-5)
    criterion = nn.MSELoss()

    training_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        print()
        epoch_train_losses = []
        for i, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()

            # Calculate score of network on training data and backpropagate
            outputs = net(x.float())
            loss = criterion(outputs, y.float())
            loss.backward()

            if applying_noise:
                # Clip norm of gradient vector to be maximum length C
                nn.utils.clip_grad_norm_(net.parameters(), clipping_parameter)
                # Apply noise to each parameter gradient
                for p in net.parameters():
                    p.grad += torch.normal(torch.zeros_like(p), noise_scale * clipping_parameter)

            optimizer.step()

            epoch_train_losses.append(loss.item())

        train_average_epoch_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        print('[epoch: %d] training set loss: %.6f' %
              (epoch + 1, train_average_epoch_loss))

        # Skip plotting the first epoch as it skews our graph
        if epoch == 0:
            continue
        training_losses.append(train_average_epoch_loss)

        # Test on test set
        with torch.no_grad():
            epoch_test_losses = []
            for i, (x, y) in enumerate(test_dataloader):
                # Calculate score of network on test dataset
                outputs = net(x.float())
                loss = criterion(outputs, y.float())

                # print statistics
                epoch_test_losses.append(loss.item())
            test_average_epoch_loss = sum(epoch_test_losses) / len(epoch_test_losses)
            print('[epoch: %d] test set loss: %.6f' %
                  (epoch + 1, test_average_epoch_loss))
            test_losses.append(test_average_epoch_loss)

            # Plot losses
            plt.close()
            plt.plot(range(len(training_losses)), training_losses, label='train_loss')
            plt.plot(range(len(test_losses)), test_losses, label='test_loss')
            plt.title('Losses as models are trained')
            plt.xlabel('Number of Epochs')
            plt.ylabel('Loss value')
            plt.legend(loc='upper right')
            # plt.savefig(path + f'/training.png')

    X_unscaled = df_fakes.values

    # Calculate scores and sanitise
    output_scores = net(torch.Tensor(X_unscaled)).detach().numpy()
    output_scores_df = pd.DataFrame(output_scores, columns=['week1', 'week2', 'week3'])
    output_scores_max_one_df = output_scores_df.applymap(lambda x: 1 if x > 1 else x)
    output_scores_max_one_min_zero_df = output_scores_max_one_df.applymap(lambda x: 0 if x < 0 else x)

    # Save results
    df_output = pd.concat([df_fakes, output_scores_max_one_min_zero_df], axis=1)
    return df_output




