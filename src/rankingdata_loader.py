import os
import pandas as pd
import logging

# ranking_result df format: User_id \t [ranked item_ids] \t [scores] \t [labels]

class RankingDataLoader(object):
    def __init__(self, ranking_result, sep='\t', seq_sep=',', label='label', g1_user_list=[],
                 g2_user_list=[]):
        self.rank_df = None
        self.sep = sep
        self.seq_sep = seq_sep
        self.label = label
        self.rank_df = ranking_result
        self.g1_user_list = g1_user_list
        self.g2_user_list = g2_user_list
        self.g1_df, self.g2_df = self._load_groups()

    def _load_groups(self):
        """
        Load advantaged/disadvantaged group info file and split the all data dataframe
        into two group-dataframes
        :return: group 1 dataframe (advantaged), group 2 dataframe (disadvantaged)
        """
        if self.rank_df is None:
            self._load_data()
        group_1_df = self.rank_df[self.rank_df['uid'].isin(self.g1_user_list)]
        group_2_df = self.rank_df[self.rank_df['uid'].isin(self.g2_user_list)]
        return group_1_df, group_2_df