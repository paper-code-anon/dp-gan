import pandas as pd
import numpy as np
from k_means_constrained import KMeansConstrained
import matplotlib.pyplot as plt

def apply_team_id_with_kmeans(df_original):
    scores_df = df_original[['week1', 'week2', 'week3']]
    scores_vals = scores_df.values

    n_clusters = round(len(df_original) / 6.5)
    kmeans = KMeansConstrained(n_clusters=n_clusters, size_min=6, size_max=7, random_state=0).fit(scores_vals)

    df = df_original.copy(deep=True)
    df['team_id'] = kmeans.labels_
    df['week1'] = df['team_id'].apply(lambda x: kmeans.cluster_centers_[x, 0])
    df['week2'] = df['team_id'].apply(lambda x: kmeans.cluster_centers_[x, 1])
    df['week3'] = df['team_id'].apply(lambda x: kmeans.cluster_centers_[x, 2])
    df = df[['team_id'] + df.drop(columns=['team_id']).columns.tolist()]
    # df.to_csv('fakes_with_labels.csv', index=False)

    plt.hist(np.bincount(kmeans.labels_), bins=[x + 0.5 for x in range(9)])
    plt.title('Histogram of team sizes')
    plt.xlabel('Team size')
    plt.ylabel('Count')
    # plt.savefig(f'team_sizes_{n_clusters}.png')
    plt.close()
    return df
