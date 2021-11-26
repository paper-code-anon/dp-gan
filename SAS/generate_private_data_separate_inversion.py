from activities_LA_paired_loss_perturbation_no_inversion import run_gan
# from convert_LA_to_scores import convert_la_to_scores
# from k_means_scores import apply_team_id_with_kmeans
# from apply_threshold import apply_threshhold_to_df
from apply_private_inverse_transformation import apply_private_inverse_transformation_fun
from calculate_noise_scale import get_noise_scale
import pandas as pd
import os
import datetime
from shutil import copy


# for dp_gan_epsilon_budget in [1, 2, 5, 10, 20, 100]:
# Privacy inputs
"""Change these"""
delta = 0.001
# dp_gan_epsilon_budget = 2
inverse_transformation_epsilon_budget = 1
# threshhold_gen_epsilon_budget = 1
# scores_gen_epsilon_budget = 6

for _ in range(10):
    for dp_gan_epsilon_budget in [2]:
    
        
        # GAN privacy inputs
        gan_batch_size = 4
        gan_num_epochs = 250
        
        # Score Generator inputs
        scores_gen_batch_size = 4
        scores_gen_num_epochs = 20
        
        # File management
        # Set up new directory for this execution. This will create a new folder with all the relevant saved content
        # from this execution script. It will save a copy of this script, so you know which script was used to generate it
        starting_time = '_'.join(str(datetime.datetime.now()).replace(':', '_').split())
        # Try to make the new directory
        path = f'gan_full_workflow_{starting_time}'
        try:
            os.stat(path)
        except:
            os.mkdir(path)
        # Copy a timestamped version of this file into the new directory
        copy(__file__, path + f'/_{starting_time}.py')
        
        
        # Calculate gan noise scale based on input parameters
        gan_df = pd.read_csv('transform_dataset_to_time_series_student_activity_withLA_withScore_v8.csv')
        gan_input_length = len(gan_df)
        gan_noise_scale = get_noise_scale(N=gan_input_length,
                                          num_epochs=gan_num_epochs,
                                          batch_size=gan_batch_size,
                                          delta=delta,
                                          input_epsilon=dp_gan_epsilon_budget)
        
        # Run the GAN to generate time series sequences
        fake_df, Gaussian_Scalings = run_gan(df=gan_df,
                          batch_size=gan_batch_size,
                          num_epochs=gan_num_epochs,
                          discriminator_noise_scale=gan_noise_scale)
        fake_df.to_csv(path + f'/gan_time_series_eps_{dp_gan_epsilon_budget}.csv', header=True, index=False)
        
        # Convert gaussian scaling object to be saved
        means_df = pd.DataFrame([scaling.mean_ for scaling in Gaussian_Scalings])
        scales_df = pd.DataFrame([scaling.scale_ for scaling in Gaussian_Scalings])
        means_df.to_csv(path + f'/means_df.csv', header=True, index=False)
        scales_df.to_csv(path + f'/scales_df.csv', header=True, index=False)
        
        means_df_read = pd.read_csv(path + f'/means_df.csv')
        scales_df_read = pd.read_csv(path + f'/scales_df.csv')
        
        fake_df_with_inverse_transform = apply_private_inverse_transformation_fun(fake_df, means_df_read, scales_df_read, inverse_transformation_epsilon_budget)
        fake_df_with_inverse_transform.to_csv(path + f'/gan_time_series_eps_{dp_gan_epsilon_budget}_with_inversetransform_eps_{inverse_transformation_epsilon_budget}.csv', header=True, index=False)
