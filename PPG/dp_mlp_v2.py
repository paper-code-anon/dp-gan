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
from sklearn.preprocessing import StandardScaler


class Classifier(nn.Module):
    def __init__(self, nx, ny):
        super(Classifier, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nx, 256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(256, ny),
        )

    def forward(self, input):
        return self.main(input)


class AccelDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def get_noise_scale(N, num_epochs, batch_size, delta, input_epsilon):
    c_2 = 1.5
    q = batch_size / N
    T = num_epochs / q
    noise_scale = c_2 * q * np.sqrt(T * np.log(1 / delta)) / input_epsilon
    return noise_scale


def reverse_transform_noisy(sample, means, scales, inverse_transformation_sensitivity, N, inverse_eps):
    noisy_mean = means + \
                 np.random.laplace(np.zeros_like(means), (inverse_transformation_sensitivity / N / inverse_eps))
    noisy_scale = np.sqrt(np.abs(scales ** 2 +
                          np.random.laplace(np.zeros_like(scales), (inverse_transformation_sensitivity ** 2 / N / inverse_eps))))
    noisy_inverse = sample * noisy_scale + noisy_mean
    return noisy_inverse


def convert_data_to_averaged(full_syn_data):
    df_data = full_syn_data[full_syn_data.columns[0:400]]
    df_columns = df_data.columns

    mean_window = 10
    means = []
    for i in range(400 // mean_window):
        means.append(df_data[df_columns[i * mean_window: (i + 1) * mean_window]].mean(axis=1))

    output_columns = [col.split('.')[0] + f'{i}-{i + mean_window}' for i, col in enumerate(df_columns[0:100]) if
                      i % mean_window == 0] + \
                     [col.split('.')[0] + f'{i}-{i + mean_window}' for i, col in enumerate(df_columns[100:200]) if
                      i % mean_window == 0] + \
                     [col.split('.')[0] + f'{i}-{i + mean_window}' for i, col in enumerate(df_columns[200:300]) if
                      i % mean_window == 0] + \
                     [col.split('.')[0] + f'{i}-{i + mean_window}' for i, col in enumerate(df_columns[300:400]) if
                      i % mean_window == 0]

    output_df = pd.concat(means, axis=1)
    output_df.columns = output_columns

    return output_df


batch_size = 16
num_epochs = 150
plot_increment = 10
N = 3038  # TODO: Note: hard coded for convenience

training_all = True  # TODO: CHeck! Whether to merge the test and training set for producing final output
applying_scaling = True

# Noise application parameters
applying_noise = True
clipping_parameter = 0.1
delta = 0.001
inverse_transformation_sensitivity = 60000
inverse_transform_eps = 1

for _ in range(1):
    for epsilon in [7]: #[10000, 100, 50, 25, 10, 5, 1]:

        if applying_noise:
            noise_scale = get_noise_scale(N, num_epochs, batch_size, delta, epsilon)


        # # 0) File and directory management
        # Set up new directory for this execution. This will create a new folder with all the relevant saved content
        # from this execution script. It will save a copy of this script, so you know which script was used to generate it
        starting_time = '_'.join(str(datetime.datetime.now()).replace(':', '_').split())
        # Try to make the new directory
        path = f'score_classifier_{starting_time}'
        try:
            os.stat(path)
        except:
            os.mkdir(path)
        # Copy a timestamped version of this file into the new directory
        copy(__file__, path + f'/_{starting_time}.py')

        # 1) Load data
        # df = pd.read_csv('accel_x_y_z+ppgv3_removed_nulls_balanced_grouped.csv')
        df = pd.read_csv('accel_x_y_z+ppgv3_removed_nulls_balanced.csv')

        # Store inverse label mapping
        activity_label_mapping = {
            1: 0,
            2: 1,
            3: 2
        }
        # Store inverse label mapping
        label_activity_mapping = {v: k for k, v in activity_label_mapping.items()}

        activity_label = df['activity'].apply(lambda x: activity_label_mapping[x]).values
        df = df.drop(columns=['activity', 'subject'])
        # df = df.drop(columns=['activity'])
        X_unscaled = df.values

        Scaling = StandardScaler()
        if applying_scaling:
            X = Scaling.fit_transform(X_unscaled)
        else:
            X = X_unscaled

        # Split data into test and training set
        X_train, X_test, y_train, y_test = train_test_split(X, activity_label, test_size=0.2, train_size=0.8, random_state=32)
        if training_all:
            X_train = X
            y_train = activity_label

        # # Manage imbalanced data
        # counts = np.bincount(y_train)
        # labels_weights = 1. / counts
        # weights = labels_weights[y_train]
        # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights), replacement=True)
        #
        # # For privacy reasons, we can only consider 1 epoch as sampled with the smallest class counts
        # allowed_samples_per_epoch = min(counts) * len(counts)

        train_dataloader = torch.utils.data.DataLoader(
            dataset=AccelDataset(X_train, y_train),
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            # sampler=sampler
        )

        test_dataloader = torch.utils.data.DataLoader(
            dataset=AccelDataset(X_test, y_test),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )

        # Set up classifier and optimizer
        nx = len(X_train[0])
        ny = 3
        net = Classifier(nx, ny)
        # net = LSTMClassifier(nx, ny)
        optimizer = optim.Adam(net.parameters(), lr=0.0001)  # Appears very sensitive to learning rate need 0.0001
        # optimizer = optim.Adam(net.parameters(), lr=0.00001, eps=1e-6, betas=(0.5, 0.999), weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        training_losses = []
        test_losses = []
        test_accuracies = []
        for epoch in range(num_epochs):
            print()
            current_epoch_count = 0

            epoch_train_losses = []
            for i, (x, y) in enumerate(train_dataloader):
                # # Check if we have already viewed our privacy-allowed sample count for a given epoch
                # current_epoch_count += len(x)
                # if current_epoch_count > allowed_samples_per_epoch:
                #     break

                optimizer.zero_grad()

                # Calculate score of network on training data and backpropagate
                outputs = net(x.float())
                loss = criterion(outputs, y)
                loss.backward()

                # # Clip norm of gradient vector to be maximum length C
                nn.utils.clip_grad_norm_(net.parameters(), clipping_parameter)

                if applying_noise:
                    # Apply noise to each parameter gradient
                    for p in net.parameters():
                        # Add noise
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
                total_correct = []
                class_count_dict = {0: 0, 1: 0, 2: 0}
                true_class_count_dict = {0: 0, 1: 0, 2: 0}

                for i, (x, y) in enumerate(test_dataloader):
                    # Calculate score of network on test dataset
                    outputs = net(x.float())
                    loss = criterion(outputs, y)

                    # print statistics
                    epoch_test_losses.append(loss.item())

                    predicted_class = torch.argmax(outputs.float(), dim=1)
                    correct = predicted_class == y
                    correct_in_batch = sum(correct).item()

                    batch_accuracy = correct_in_batch / batch_size
                    total_correct.append(batch_accuracy)

                    for prediction in predicted_class:
                        class_count_dict[prediction.item()] += 1

                    for true_class in y:
                        true_class_count_dict[true_class.item()] += 1

                test_average_epoch_loss = sum(epoch_test_losses) / len(epoch_test_losses)
                print('[epoch: %d] test set loss: %.6f' %
                      (epoch + 1, test_average_epoch_loss))

                test_accuracy = round(sum(total_correct) / len(total_correct), 2) * 100
                print(f'Accuracy is: {test_accuracy}%')

                test_losses.append(test_average_epoch_loss)
                test_accuracies.append(test_accuracy)
                print(class_count_dict)
                print(true_class_count_dict)

                smoothed_accuracy = pd.Series(test_accuracies).iloc[:].rolling(window=5).mean()

                # if epoch % plot_increment == 0:
                # Plot losses
                plt.close()
                plt.plot(range(len(training_losses)), training_losses, label='train_loss')
                plt.plot(range(len(test_losses)), test_losses, label='test_loss')
                plt.title('Losses as models are trained')
                plt.xlabel('Number of Epochs')
                plt.ylabel('Loss value')
                plt.legend(loc='upper right')
                plt.savefig(path + f'/training.png')

                # Plot accuracies
                plt.close()
                plt.plot(range(len(test_accuracies)), test_accuracies, label='test_accuracy')
                plt.plot(range(len(smoothed_accuracy)), smoothed_accuracy, label='smoothed_accuracy')
                plt.title('Test accuracy as models are trained')
                plt.xlabel('Number of Epochs')
                plt.ylabel('Accuracy %')
                plt.legend(loc='upper right')
                plt.savefig(path + f'/accuracy.png')


        for folder_name in [
                            'mlp_dp_loss_10_runs',
                            'mlp_dp_sgd_10_runs',
                            'lstm_dp_loss_10_runs',
                            'lstm_dp_sgd_10_runs',

                            ]:
            for iteration in range(10):
                fake_vals_df = pd.read_csv(f'{folder_name}/generated_fakes_normalised_{iteration}.csv')
                # output_classes = torch.argmax(net(torch.Tensor(convert_data_to_averaged(fake_vals_df).values)), dim=1).detach().numpy()
                output_classes = torch.argmax(net(torch.Tensor(fake_vals_df.values)), dim=1).detach().numpy()
                output_classes_df = pd.DataFrame(output_classes, columns=['activity_label'])
                output_activities = output_classes_df.applymap(lambda x: label_activity_mapping[x])

                # Apply inverse noisy transform
                if applying_scaling:
                    fake_vals = np.apply_along_axis(
                        lambda m: reverse_transform_noisy(m,
                                                          Scaling.mean_,
                                                          Scaling.scale_,
                                                          inverse_transformation_sensitivity,
                                                          N,
                                                          inverse_transform_eps),
                        axis=1,
                        arr=fake_vals_df.values)
                    fake_vals_df = pd.DataFrame(fake_vals, columns=fake_vals_df.columns)

                # eps_1_df = pd.read_csv(f'{folder_name}/generated_fakes.csv')

                fake_vals_with_activity = pd.concat([fake_vals_df, output_activities], axis=1)
                # fake_vals_with_activity.to_csv(path + f'/{folder_name}_{epsilon}_classified_samples.csv', index=False)
                fake_vals_with_activity.to_csv(path + f'/{folder_name}_classified_samples_{iteration}.csv', index=False)





