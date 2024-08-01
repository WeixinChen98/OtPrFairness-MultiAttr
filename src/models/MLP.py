# coding=utf-8

import torch
import torch.nn as nn
from models.BaseRecModel import BaseRecModel
from utils.constants import *



class MLP(BaseRecModel):
    @staticmethod
    def parse_model_args(parser, model_name='MLP'):
        parser.add_argument('--num_layers', type=int, default=3,
                            help="Number of mlp layers.")
        return BaseRecModel.parse_model_args(parser, model_name)

    def __init__(self, data_processor_dict, user_num, item_num, u_vector_size, i_vector_size,
                 num_layers=3, random_seed=2020, dropout=0.2, model_path='../model/Model/Model.pt'):
        self.num_layers = num_layers
        self.factor_size = u_vector_size // (2 ** (self.num_layers - 1))
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
        # Init embeddings
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.u_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.u_vector_size)

        # Init MLP
        self.mlp = nn.ModuleList([])
        pre_size = self.factor_size * (2 ** self.num_layers)
        for i in range(self.num_layers):
            self.mlp.append(nn.Dropout(p=self.dropout))
            self.mlp.append(nn.Linear(pre_size, pre_size // 2))
            self.mlp.append(nn.ReLU())
            pre_size = pre_size // 2
        self.mlp = nn.Sequential(*self.mlp)

        # Init predictive layer
        self.p_layer = nn.ModuleList([])
        assert pre_size == self.factor_size
        # pre_size = pre_size * 2
        self.prediction = torch.nn.Linear(pre_size, 1)

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict['X'][:, 0] - 1
        i_ids = feed_dict['X'][:, 1] - 1

        mlp_u_vectors = self.uid_embeddings(u_ids)
        mlp_i_vectors = self.iid_embeddings(i_ids)

        mlp = torch.cat((mlp_u_vectors, mlp_i_vectors), dim=-1)
        mlp = self.mlp(mlp)

        prediction = self.prediction(mlp).view([-1])

        out_dict = {'prediction': prediction,
                    'check': check_list,
                    'u_vectors': mlp_u_vectors}
        return out_dict

# filter_mode = 'separate' on single or all feature(s), similar to PFRec
# Hence remove the personalized setting due to the same fairness requirement for all users
class MLP_PCFR(MLP):
    def _init_nn(self):
        MLP._init_nn(self)
        self._init_sensitive_filter()

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict['X'][:, 0] - 1
        i_ids = feed_dict['X'][:, 1] - 1

        mlp_u_vectors = self.uid_embeddings(u_ids)
        mlp_i_vectors = self.iid_embeddings(i_ids)

        mlp_u_vectors = self.filter(mlp_u_vectors)

        mlp = torch.cat((mlp_u_vectors, mlp_i_vectors), dim=-1)
        mlp = self.mlp(mlp)

        prediction = self.prediction(mlp).view([-1])

        out_dict = {'prediction': prediction,
                    'check': check_list,
                    'u_vectors': mlp_u_vectors}
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


class MLP_FairRec(MLP):
    def _init_nn(self):
        MLP._init_nn(self)
        self._init_sensitive_learner()
        self._init_sensitive_filter()

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict['X'][:, 0] - 1
        i_ids = feed_dict['X'][:, 1] - 1

        mlp_u_vectors = self.uid_embeddings(u_ids)
        mlp_i_vectors = self.iid_embeddings(i_ids)

        mlp_u_learner_vectors = self.learner(mlp_u_vectors)
        mlp_u_filter_vectors = self.filter(mlp_u_vectors)

        if self.training:
            mlp_u_vectors = mlp_u_learner_vectors + mlp_u_filter_vectors            
        else:
            mlp_u_vectors = mlp_u_filter_vectors

        mlp = torch.cat((mlp_u_vectors, mlp_i_vectors), dim=-1)
        mlp = self.mlp(mlp)

        prediction = self.prediction(mlp).view([-1])

        out_dict = {'prediction': prediction,
                    'check': check_list,
                    'u_vectors': mlp_u_vectors,
                    'u_learner_vectors': mlp_u_learner_vectors,
                    'u_filter_vectors': mlp_u_filter_vectors}
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

class MLP_FOCF_AbsUnf(MLP):
    def forward(self, feed_dict):
        out_dict = self.predict(feed_dict)
        batch_size = int(feed_dict[LABEL].shape[0] / 2)
        pos, neg = out_dict['prediction'][:batch_size], out_dict['prediction'][batch_size:]
        loss = -(pos - neg).sigmoid().log().sum()
        out_dict['loss'] = loss
        abs_loss = torch.abs(out_dict['prediction'] - feed_dict[LABEL])
        out_dict['abs_loss'] = abs_loss
        return out_dict

class MLP_FOCF_ValUnf(MLP):
    def forward(self, feed_dict):
        out_dict = self.predict(feed_dict)
        batch_size = int(feed_dict[LABEL].shape[0] / 2)
        pos, neg = out_dict['prediction'][:batch_size], out_dict['prediction'][batch_size:]
        loss = -(pos - neg).sigmoid().log().sum()
        out_dict['loss'] = loss
        val_loss = out_dict['prediction'] - feed_dict[LABEL]
        out_dict['val_loss'] = val_loss
        return out_dict

# filter_mode = 'separate' on single or all feature(s), similar to PFRec
# Hence remove the personalized setting due to the same fairness requirement for all users
class MLP_SolutionA(MLP):
    def _init_nn(self):
        MLP._init_nn(self)
        self._init_sensitive_filter()

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict['X'][:, 0] - 1
        i_ids = feed_dict['X'][:, 1] - 1

        mlp_u_vectors = self.uid_embeddings(u_ids)
        mlp_i_vectors = self.iid_embeddings(i_ids)

        mlp_u_vectors = self.filter(mlp_u_vectors)

        mlp = torch.cat((mlp_u_vectors, mlp_i_vectors), dim=-1)
        mlp = self.mlp(mlp)

        prediction = self.prediction(mlp).view([-1])

        out_dict = {'prediction': prediction,
                    'check': check_list,
                    'u_vectors': mlp_u_vectors}
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

    def forward(self, feed_dict):
        out_dict = self.predict(feed_dict)
        batch_size = int(feed_dict[LABEL].shape[0] / 2)
        pos, neg = out_dict['prediction'][:batch_size], out_dict['prediction'][batch_size:]
        loss = -(pos - neg).sigmoid().log().sum()
        out_dict['loss'] = loss
        abs_loss = torch.abs(out_dict['prediction'] - feed_dict[LABEL])
        out_dict['abs_loss'] = abs_loss
        return out_dict

    