# transform the original dataset to a form that can be accepted by the time series synthesizer

import pandas as pd
from datetime import datetime
import numpy as np
#matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import os

save_path = f'transformed_data'
try:
    os.stat(save_path)
except:
    os.mkdir(save_path)

dataset_name = 'student_activity_8_program.csv'
#dataset_name = 'student_activity_8_program.csv'
df = pd.read_csv(dataset_name, index_col=None)

auxiliary_dataset_name = 'team_scores_8_program.csv'
#auxiliary_dataset_name = 'team_scores_8_program.csv'
score_df = pd.read_csv(auxiliary_dataset_name, index_col=None)


df.columns
df.iloc[0]
df['program_id'].unique()
len(df['team_id'].unique())
len(df['student_id'].unique())
len(df['LA_label'].unique())
len(df['week'].unique())
len(df.groupby(['program_id', 'student_id']).size())


# Use real_date_format so we can sort by date
df['real_date_format'] = \
    df['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))

# Prior knowledge on the length of a program
relative_timeline_day_num = 21

# set the number of weeks
week_num = 3

# get the LA labels
LA_label_list_array = np.array(df['LA_label'].unique().astype(str), dtype=object)
LA_label_list_array = LA_label_list_array[LA_label_list_array != 'nan']
# should be 11 LA categories
LA_num = len(LA_label_list_array)

# generate the dataframe attribute names
# normal attributes followed by a time series
time_series_columns_suffix = \
    np.array(np.tile(['Day_'], relative_timeline_day_num), dtype=object) + \
    np.array(np.arange(relative_timeline_day_num).astype(str), dtype=object)

# generate the time series column names with LA prefix
time_series_columns = [x+'_'+time_series_columns_suffix for x in LA_label_list_array]
time_series_columns = np.reshape(time_series_columns, \
                                (1, LA_num*relative_timeline_day_num))[0]

# get the attribute columns
trans_data_columns = ['student_id', 'program_id', 'team_id', 'start_date']
trans_data_columns.extend(time_series_columns)

# add 3 columns of weekly scores
week_score_columns = \
    np.array(np.tile(['score_week'], week_num), dtype=object) + \
    np.array((np.arange(week_num)+1).astype(str), dtype=object)
trans_data_columns.extend(week_score_columns)

# get the columns of time series and scores
time_and_score_columns = list(time_series_columns)
time_and_score_columns.extend(week_score_columns)



# initialize the dataframe of the transformed data
trans_data = pd.DataFrame(0, \
                          index=np.arange(len(df.groupby(['program_id', 'student_id']).size())), \
                          columns=trans_data_columns)

# initialize the program_id to -999 so that we can remove abnormal records later
trans_data['program_id'] = -999
trans_data.columns

# get the unique program ids
unique_programs = df['program_id'].unique()

# initialize the student time series, should be a matrix, actually
student_time_series_array = []

# the number of programs having a running period larger than [relative_timeline_day_num] days
abnormal_program_num = 0

# the loc index of the transformed dataset
trans_data_idx = 0

for program_id in unique_programs:
    print('\nProcessing program_id: {} ......'.format(program_id))
    
    # get the sub-dataframe for this program id
    program_id_df = df[df['program_id'] == program_id]
    
    # Find unique dates for this program
    program_activity_counts = program_id_df.groupby(['real_date_format']).size()
    # sort the unique dates and get the time length in terms of days
    unique_dates = program_activity_counts.index
    
    first_date = unique_dates[0]
    last_date = unique_dates[-1]
    
    diff_day_num = (last_date-first_date).days+1
    
    if diff_day_num != relative_timeline_day_num:
        abnormal_program_num = abnormal_program_num + 1
        print('\n{0}. Abnormal program ID: {1}'.format(abnormal_program_num, program_id))
        print('Program duration is: {}'.format(diff_day_num))
    
#    first_date = min(program_id_df['real_date_format'].unique())
#    last_date = max(program_id_df['real_date_format'].unique())
    
    # get the program-specific running dates
    # for simplicity, the program will be cut off on the 21st day 
    program_dates = [first_date + np.timedelta64(x, 'D') for x in range(relative_timeline_day_num)]

    # Generate an empty histogram of dates for each program
    unique_dates_df = pd.DataFrame({'zero_interactions': np.zeros(len(program_dates))}, index=program_dates)

    # find the students who participate in this program
    unique_students = program_id_df['student_id'].unique()
    
    for student_id in unique_students:
        # Select the dataframe for a given student and a program id
        student_program_dates_df = program_id_df[program_id_df['student_id'] == student_id]
        
        # get the weekly scores for week1 to week3 sequentially
        sub_score_df = score_df[(score_df['team_id']==student_program_dates_df.iloc[0]['team_id']) & (score_df['program_id']==program_id)]
        sorted_sub_score_df = sub_score_df.sort_values(by='week', ascending=True)
        
        # Some data is missing, if so skip this
        if (len(sorted_sub_score_df['moderated_score.max'].values) == week_num) and (sum(sorted_sub_score_df['moderated_score.max'].values) > 0):
            trans_data.loc[trans_data_idx, week_score_columns] = sorted_sub_score_df['moderated_score.max'].values
        else:
            print('\ndeleting an abnormal record...')
            print('program id: {}'.format(program_id))
            print('team id: {}'.format(student_program_dates_df.iloc[0]['team_id']))
            print('student id: {}'.format(student_id))
            print('questionable weekly scores:')
            print(sorted_sub_score_df['moderated_score.max'].values)
            #print(sum(sorted_sub_score_df['moderated_score.max'].values))
            #print(sum(sorted_sub_score_df['moderated_score.max'].values) > 0)
            continue
        
        # update the transformed dataframe
        trans_data.loc[trans_data_idx, 'student_id'] = student_id
        trans_data.loc[trans_data_idx, 'program_id'] = program_id
        trans_data.loc[trans_data_idx, 'team_id'] = student_program_dates_df.iloc[0]['team_id']
        trans_data.loc[trans_data_idx, 'start_date'] = first_date
        
        # check LA one by one
        for this_LA in LA_label_list_array:
            
            # get the sub-dataframe for a given LA and a given student in a given program/team
            LA_student_program_dates_df = student_program_dates_df[student_program_dates_df['LA_label'] == this_LA]
            # print(len(LA_student_program_dates_df))
            
            # for non-empty sub-dataframes
            if len(LA_student_program_dates_df) > 0:
                # Group student interactions by date, extracting counts and sort in order
                histo_df = LA_student_program_dates_df['real_date_format'].value_counts(sort=False).sort_index()
        
                # Join our histogram to our template to fill out any missing dates.
                # If the program is longer than the relative_timeline_day_num, it will be truncated by this join
                filled_out_df = unique_dates_df.join(histo_df)
                LA_student_time_series = (filled_out_df['zero_interactions'] + filled_out_df['real_date_format']).fillna(0).values
            
                trans_data.loc[trans_data_idx, \
                               this_LA+'_'+time_series_columns_suffix] = \
                               LA_student_time_series
            # sanity check: the following equaiton should hold
            # be sure to exclude the records with "LA_label = nan"
            # sum(trans_data.loc[trans_data_idx, time_series_columns]) == sum(student_program_dates_df.groupby(['LA_label']).size())
        
        # update the matrix of the student time series
        # student_time_series_array.append(np.array(trans_data.loc[trans_data_idx, time_series_columns]).astype(float))
        student_time_series_array.append(np.array(trans_data.loc[trans_data_idx, time_and_score_columns]).astype(float))
        
        # finish the dataframe update
        trans_data_idx = trans_data_idx + 1


# write the matrix of the student time series to a csv file
# df_output = pd.DataFrame(student_time_series_array, columns=time_series_columns)
df_output = pd.DataFrame(student_time_series_array, columns=time_and_score_columns)
df_output.to_csv(save_path+'/'+'ev_transformed_activities_withLA_withScore_withHeader.csv', index=False, header=True)
df_output.to_csv(save_path+'/'+'ev_transformed_activities_withLA_withScore_noHeader.csv', index=False, header=False)

# write the dataframe to a csv file to be process by RNN GAN
# Some students are skipped due to empty scores, drop these rows
trans_data = trans_data[trans_data['program_id'] != -999]
trans_data.to_csv(save_path+'/'+'transform_dataset_to_time_series_student_activity_withLA_withScore_v8.csv', index=False)

# test the output csv file
test_data = pd.read_csv(save_path+'/'+'transform_dataset_to_time_series_student_activity_withLA_withScore_v8.csv')    
test_data.iloc[0]

len(test_data)
test_data['program_id'].unique()
len(test_data['team_id'].unique())
len(test_data['student_id'].unique())



# generate fake score data
generate_fake_score_df_enabler = 0
if generate_fake_score_df_enabler == 1:
    
    import numpy.matlib
    
    unique_team_program_columns = trans_data.groupby(['team_id','program_id']).size().reset_index().rename(columns={0:'count'})
    unique_team_program_num = len(unique_team_program_columns)
    
    team_id_column = \
        np.reshape(np.matlib.repmat(unique_team_program_columns[['team_id']].values, 1, week_num).T, -1, 1)
    program_id_column = \
        np.reshape(np.matlib.repmat(unique_team_program_columns[['program_id']].values, 1, week_num).T, -1, 1)
    
    week_column = \
        np.array(np.tile(['week'], week_num), dtype=object) + \
        np.array((np.arange(week_num)+1).astype(str), dtype=object)
    week_column = np.matlib.repmat(week_column, 1, unique_team_program_num)[0]
    
    assessment_type_column = np.matlib.repmat(['score_plan', 'score_draft', 'score_report'], 1, unique_team_program_num)[0]
    
    fake_score_df = pd.DataFrame(np.array([team_id_column, program_id_column, assessment_type_column, week_column]).T, \
                                 columns=['team_id', 'program_id', 'assessment_type', 'week'])
    fake_score_df['score'] = np.random.rand(week_num*unique_team_program_num)
    fake_score_df.to_csv('student_fake_score_8_program.csv', index=False, header=True)





fig, axes = plt.subplots(figsize=(40,6))

for student_idx in range(10):
    plt.plot(time_series_columns, \
             trans_data.loc[student_idx, time_series_columns], \
             label=trans_data.loc[student_idx, 'student_id'])

plt.xlabel('Time with LA Labels')
plt.ylabel('Daily acitivty counts')
title_str = 'Student daily activity counts for '+str(LA_num)+' LA labels'
plt.title(title_str)
plt.xticks(rotation=90)
plt.legend(loc='best')





# visualize individual time series with LA labels
fig, axes = plt.subplots(figsize=(40,25))
#plt.title('Illustration of the original samples ('+str(64)+' samples)')

for this_LA_idx in range(LA_num):
    plt.subplot(3, 4, this_LA_idx+1)
    
    this_LA_label = LA_label_list_array[this_LA_idx]
    plt.title(this_LA_label)
    plt.xticks(rotation=90)
    plt.xlabel('Time')
    plt.ylabel('Daily acitivty counts')
    
    for student_idx in range(10):
        plt.plot(time_series_columns_suffix, \
                 trans_data.loc[student_idx, time_series_columns[this_LA_idx*relative_timeline_day_num : (this_LA_idx+1)*relative_timeline_day_num]], \
                 label=trans_data.loc[student_idx, 'student_id'])
        
    plt.legend(loc='best')







# initialize the cross correlation matrix for the LA time series
LA_time_series_cross_corr_matrix = pd.DataFrame(np.zeros((LA_num, LA_num)), \
                                         columns=LA_label_list_array, \
                                         index=LA_label_list_array)

# check the LA time series for each student
for student_idx in range(len(trans_data)):
    # get the LA time series for this student
    this_row = trans_data.iloc[student_idx][time_series_columns]
    # reshape the LA time series to 21*11
    reshape_this_row = np.reshape(this_row.values, (relative_timeline_day_num, LA_num)).astype(float)
    # add the LA labels to the LA time series
    reshape_this_row_df = pd.DataFrame(reshape_this_row, columns=LA_label_list_array)
    # perturb the LA time series a little bit to remove zero values
    reshape_this_row_df = reshape_this_row_df + np.random.uniform(-1, 1, (relative_timeline_day_num, LA_num)) * 1e-1
    # calculate the cross correlation matrix
    reshape_this_row_df_cov = (reshape_this_row_df).corr(method='pearson')
    
    # udpate the cross correlation matrix for LA time series
    LA_time_series_cross_corr_matrix = LA_time_series_cross_corr_matrix + reshape_this_row_df_cov
    
# normalize the cross correlation matrix for the LA time series
LA_time_series_cross_corr_matrix = LA_time_series_cross_corr_matrix/len(trans_data)    
    
    
# the cross correlation matrix for the LA time series
figure_LA_time_series_cross_corr = plt.figure(figsize=(10,10))
figure_handler = sns.heatmap(LA_time_series_cross_corr_matrix, \
                             cmap='jet', vmin=-1, vmax=1, \
                             xticklabels=reshape_this_row_df_cov.columns, \
                             yticklabels=reshape_this_row_df_cov.columns, annot=False)

title_str = 'Normalized covariance matrix of the LA time series data'

figure_handler.set_title(title_str, size=14)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.savefig(save_path+'/'+'figure_LA_time_series_cross_corr.png', format='png', dpi=200)

    
    


from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from matplotlib import pyplot as plt

# Perform clustering, you can choose the method
# in this case, we use 'ward'
Z = linkage(LA_time_series_cross_corr_matrix, 'ward')

# Extract the membership to a cluster, either specify the n_clusters
# or the cut height
# (similar to sklearn labels)
print(cut_tree(Z, n_clusters=3))

# Visualize the clustering as a dendogram
fig = plt.figure(figsize=(5, 5))
dn = dendrogram(Z, orientation='top', labels=LA_time_series_cross_corr_matrix.columns)
plt.xticks(rotation=90)
plt.show()

