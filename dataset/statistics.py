import os
import numpy as np
import pandas as pd

datasets = ['ml1M', 'insurance']
for dataset in datasets:
    print(dataset)
    data_directory = './' + dataset
    all_df = pd.read_csv(os.path.join(data_directory, dataset + '.all.tsv'),
                        header=0, sep='\t', engine='python')
    u_count = len(all_df['uid'].unique())
    i_count = len(all_df['iid'].unique())
    density = float(all_df.shape[0]) / float(u_count) / float(i_count)
    print('interactions: ' + str(all_df.shape[0]))
    print('users: ' + str(u_count))
    print('items: ' + str(i_count))
    print('density: ' + str(density))

