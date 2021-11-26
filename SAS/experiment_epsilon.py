# from activities_LA_paired_loss_perturbation_no_inversion import run_gan
from convert_LA_to_scores import convert_la_to_scores
from k_means_scores import apply_team_id_with_kmeans
from apply_threshold import apply_threshhold_to_df
from calculate_noise_scale import get_noise_scale
from apply_private_inverse_transformation import apply_private_inverse_transformation_fun
from check_utility import save_utility_image
import pandas as pd
import os
import datetime
from shutil import copy


"""Input variables: Only change these!"""
#####################################################################################
dp_gan_folder = 'gan_benchmark_eps_2'
dp_gan_filename = 'gan_time_series_eps_2.csv'

threshhold_gen_epsilon_budget = 1
inverse_transformation_epsilon_budget = 10
scores_gen_epsilon_budget = 10
##################################################################################

delta = 0.001

# Score Generator variables
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

# Calculate scores gen noise scale based on input parameters
scores_gen_df = gan_df[['score_week1', 'score_week2', 'score_week3']].dropna()
scores_gen_length = len(scores_gen_df)
score_gen_noise_scale = get_noise_scale(N=scores_gen_length,
                                        num_epochs=scores_gen_num_epochs,
                                        batch_size=scores_gen_batch_size,
                                        delta=delta,
                                        input_epsilon=scores_gen_epsilon_budget)

# Load df from folder
fake_df = pd.read_csv(dp_gan_folder + '/' + dp_gan_filename)
dp_gan_filename_str = dp_gan_filename.strip('.csv')
# Load mean and variance details from folder
means_df_read = pd.read_csv(dp_gan_folder + f'/means_df.csv')
scales_df_read = pd.read_csv(dp_gan_folder + f'/scales_df.csv')

# Apply private inverse transformation to data
fake_df_with_inverse_transform = apply_private_inverse_transformation_fun(fake_df, means_df_read, scales_df_read,
                                                                          inverse_transformation_epsilon_budget)
fake_df_with_inverse_transform.to_csv(
    path + f'/{dp_gan_filename_str}_with_inversetransform_eps_{inverse_transformation_epsilon_budget}.csv')

# Generate scores for each given time series
fake_df_with_scores = convert_la_to_scores(df=gan_df,
                                           df_fakes=fake_df_with_inverse_transform,
                                           batch_size=scores_gen_batch_size,
                                           num_epochs=scores_gen_num_epochs,
                                           noise_scale=score_gen_noise_scale)
fake_scores = fake_df_with_scores[['week1', 'week2', 'week3']]
fake_df_with_scores.to_csv(path + f'/{dp_gan_filename_str}_with_inversetransform_eps_{inverse_transformation_epsilon_budget}_with_scores_eps_{scores_gen_epsilon_budget}.csv', header=True, index=False)

# Apply the threshhold cutoff to make time series sequences more realistic
fake_df_with_threshhold = apply_threshhold_to_df(original_df=gan_df,
                                                 new_df=fake_df_with_inverse_transform,
                                                 threshhold_epsilon=threshhold_gen_epsilon_budget)
# Tape the threshhold dataframe to the scores dataframe and save
fake_df_with_scores_with_threshhold = pd.concat([fake_df_with_threshhold, fake_scores], axis=1)
fake_df_with_scores_with_threshhold.to_csv(path + f'/{dp_gan_filename_str}_with_inversetransform_eps_{inverse_transformation_epsilon_budget}_with_scores_eps_{scores_gen_epsilon_budget}_with_threshold_eps_{threshhold_gen_epsilon_budget}.csv', header=True, index=False)


# Generate teamids based on the scores
fake_df_with_scores_with_threshhold_with_teamids = apply_team_id_with_kmeans(fake_df_with_scores_with_threshhold)

# Save results to CSV
print(fake_df_with_scores_with_threshhold_with_teamids)
final_name = f'/{dp_gan_filename_str}_with_inversetransform_eps_{inverse_transformation_epsilon_budget}_with_scores_eps_{scores_gen_epsilon_budget}__with_threshold_eps_{threshhold_gen_epsilon_budget}_with_teamids.csv'
fake_df_with_scores_with_threshhold_with_teamids.to_csv(
    path + final_name, header=True, index=False)

# Save utility image
save_utility_image(gan_df, path, final_name)