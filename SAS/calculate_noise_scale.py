import numpy as np


def get_noise_scale(N, num_epochs, batch_size, delta, input_epsilon):
    # Ref: Deep Learning with Differential Privacy https://arxiv.org/abs/1607.00133
    c_2 = 1.5  # Constant from google paper
    q = batch_size / N
    T = num_epochs / q
    noise_scale = c_2 * q * np.sqrt(T * np.log(1 / delta)) / input_epsilon
    return noise_scale