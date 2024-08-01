# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseRecModel import BaseRecModel
from utils.constants import *


class DMF(BaseRecModel):
    @staticmethod
    def parse_model_args(parser, model_name='DMF'):
        parser.add_argument('--num_layers', type=int, default=3,
                            help="Number of mlp layers.")
        return BaseRecModel.parse_model_args(parser, model_name)

    def __init__(self, data_processor_dict, user_num, item_num, u_vector_size, i_vector_size,
                 num_layers=3, random_seed=2020, dropout=0.2, model_path='../model/Model/Model.pt'):
        self.num_layers = num_layers
        BaseRecModel.__init__(self, data_processor_dict, user_num, item_num, u_vector_size, i_vector_size,
                              random_seed=random_seed, dropout=dropout, model_path=model_path)

    @staticmethod
    def init_weights(m):
        """
        initialize nn weights，called in main.py
        :param m: parameter or the nn
        :return:
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, a=1, nonlinearity='sigmoid')
            if m.bias is not None:
                m.bias.data.zero_()
        elif type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def _init_nn(self):
        self.uid_embeddings = nn.Embedding(self.user_num, self.u_vector_size)
        self.iid_embeddings = nn.Embedding(self.item_num, self.u_vector_size)

        self.cos = nn.CosineSimilarity()

        self.u_mlp = nn.ModuleList([nn.Linear(self.u_vector_size, self.u_vector_size)])
        for layer in range(self.num_layers - 1):
            self.u_mlp.append(nn.Linear(self.u_vector_size, self.u_vector_size))
        self.i_mlp = nn.ModuleList([nn.Linear(self.u_vector_size, self.u_vector_size)])
        for layer in range(self.num_layers - 1):
            self.i_mlp.append(nn.Linear(self.u_vector_size, self.u_vector_size))

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict['X'][:, 0] - 1
        i_ids = feed_dict['X'][:, 1] - 1

        user_embeddings = self.uid_embeddings(u_ids)
        item_embeddings = self.iid_embeddings(i_ids)
        u_input = user_embeddings

        for layer in self.u_mlp:
            u_input = layer(u_input)
            u_input = F.relu(u_input)
            u_input = torch.nn.Dropout(p=self.dropout)(u_input)

        i_input = item_embeddings
        for layer in self.i_mlp:
            i_input = layer(i_input)
            i_input = F.relu(i_input)
            i_input = torch.nn.Dropout(p=self.dropout)(i_input)

        prediction = self.cos(u_input, i_input).view([-1]) * 10
        out_dict = {'prediction': prediction,
                    'check': check_list,
                    'u_vectors': u_input}
        return out_dict


class DMF_PCFR(DMF):
    def _init_nn(self):
        DMF._init_nn(self)
        self._init_sensitive_filter()

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict['X'][:, 0] - 1
        i_ids = feed_dict['X'][:, 1] - 1

        user_embeddings = self.uid_embeddings(u_ids)
        item_embeddings = self.iid_embeddings(i_ids)

        u_input = self.filter(user_embeddings)

        for layer in self.u_mlp:
            u_input = layer(u_input)
            u_input = F.relu(u_input)
            u_input = torch.nn.Dropout(p=self.dropout)(u_input)

        i_input = item_embeddings
        for layer in self.i_mlp:
            i_input = layer(i_input)
            i_input = F.relu(i_input)
            i_input = torch.nn.Dropout(p=self.dropout)(i_input)

        prediction = self.cos(u_input, i_input).view([-1]) * 10
        out_dict = {'prediction': prediction,
                    'check': check_list,
                    'u_vectors': u_input}
        return out_dict

    def _init_sensitive_filter(self):
        def get_sensitive_filter(embed_dim):
            sequential = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.LeakyReLU(),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(embed_dim)
            )
            return sequential
        self.filter = get_sensitive_filter(self.u_vector_size)


class DMF_FairRec(DMF):
    def _init_nn(self):
        DMF._init_nn(self)
        self._init_sensitive_learner()
        self._init_sensitive_filter()

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict['X'][:, 0] - 1
        i_ids = feed_dict['X'][:, 1] - 1

        user_embeddings = self.uid_embeddings(u_ids)
        item_embeddings = self.iid_embeddings(i_ids)
        
        cf_u_learner_vectors = self.learner(user_embeddings)
        cf_u_filter_vectors = self.filter(user_embeddings)

        for layer in self.u_mlp:
            cf_u_learner_vectors = layer(cf_u_learner_vectors)
            cf_u_learner_vectors = F.relu(cf_u_learner_vectors)
            cf_u_learner_vectors = torch.nn.Dropout(p=self.dropout)(cf_u_learner_vectors)

        for layer in self.u_mlp:
            cf_u_filter_vectors = layer(cf_u_filter_vectors)
            cf_u_filter_vectors = F.relu(cf_u_filter_vectors)
            cf_u_filter_vectors = torch.nn.Dropout(p=self.dropout)(cf_u_filter_vectors)

        if self.training:
            u_input = cf_u_learner_vectors + cf_u_filter_vectors            
        else:
            u_input = cf_u_filter_vectors

        i_input = item_embeddings
        for layer in self.i_mlp:
            i_input = layer(i_input)
            i_input = F.relu(i_input)
            i_input = torch.nn.Dropout(p=self.dropout)(i_input)

        prediction = self.cos(u_input, i_input).view([-1]) * 10
        out_dict = {'prediction': prediction,
            'check': check_list,
            'u_vectors': u_input,
            'u_learner_vectors': cf_u_learner_vectors,
            'u_filter_vectors': cf_u_filter_vectors}
        return out_dict

    def _init_sensitive_learner(self):
        def get_sensitive_learner(embed_dim):
            sequential = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.LeakyReLU(),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(embed_dim)
            )
            return sequential
        self.learner = get_sensitive_learner(self.u_vector_size)

    def _init_sensitive_filter(self):
        def get_sensitive_filter(embed_dim):
            sequential = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.LeakyReLU(),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(embed_dim)
            )
            return sequential
        self.filter = get_sensitive_filter(self.u_vector_size)


class DMF_FOCF_AbsUnf(DMF):
    def forward(self, feed_dict):
        out_dict = self.predict(feed_dict)
        batch_size = int(feed_dict[LABEL].shape[0] / 2)
        pos, neg = out_dict['prediction'][:batch_size], out_dict['prediction'][batch_size:]
        loss = -(pos - neg).sigmoid().log().sum()
        out_dict['loss'] = loss
        abs_loss = torch.abs(out_dict['prediction'] - feed_dict[LABEL])
        out_dict['abs_loss'] = abs_loss
        return out_dict

class DMF_FOCF_ValUnf(DMF):
    def forward(self, feed_dict):
        out_dict = self.predict(feed_dict)
        batch_size = int(feed_dict[LABEL].shape[0] / 2)
        pos, neg = out_dict['prediction'][:batch_size], out_dict['prediction'][batch_size:]
        loss = -(pos - neg).sigmoid().log().sum()
        out_dict['loss'] = loss
        val_loss = out_dict['prediction'] - feed_dict[LABEL]
        out_dict['val_loss'] = val_loss
        return out_dict