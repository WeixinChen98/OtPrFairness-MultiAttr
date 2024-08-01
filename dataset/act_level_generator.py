import os
import numpy as np
import pandas as pd

datasets = ['ml1M', 'insurance']
file_suffix = ['.all.tsv', '.train.tsv', '.validation.tsv', '.test.tsv']
for dataset in datasets:
    print(dataset)
    data_directory = './' + dataset
    all_df = pd.read_csv(os.path.join(data_directory, dataset + '.all.tsv'),
                        header=0, sep='\t', engine='python')
    all_count_df = all_df.groupby(['uid']).size().reset_index(name='count')
    count_median = all_count_df['count'].median()

    
    all_count_df['u_activity'] = all_count_df['count'].apply(lambda x: 1 if x > count_median else 0)
    user_act_dict = dict(zip(all_count_df.uid, all_count_df.u_activity))

    for suf in file_suffix:
        print(suf)
        df = pd.read_csv(os.path.join(data_directory, dataset + suf),
                            header=0, sep='\t', engine='python')
        
        df['u_activity'] = df['uid'].map(user_act_dict)
        df.to_csv(os.path.join(data_directory, dataset + suf), sep='\t', index=False)