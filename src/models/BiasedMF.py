# coding=utf-8

import torch
from torch import nn
from models.BaseRecModel import BaseRecModel
from utils.constants import *



class BiasedMF(BaseRecModel):
    def _init_nn(self):
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.u_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.i_vector_size)
        assert self.u_vector_size == self.i_vector_size
        self.user_bias = torch.nn.Embedding(self.user_num, 1)
        self.item_bias = torch.nn.Embedding(self.item_num, 1)
        self.global_bias = torch.nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict['X'][:, 0] - 1
        i_ids = feed_dict['X'][:, 1] - 1

        # bias
        u_bias = self.user_bias(u_ids).view([-1])
        i_bias = self.item_bias(i_ids).view([-1])

        cf_u_vectors = self.uid_embeddings(u_ids)
        cf_i_vectors = self.iid_embeddings(i_ids)

        prediction = (cf_u_vectors * cf_i_vectors).sum(dim=1).view([-1])
        prediction = prediction + u_bias + i_bias + self.global_bias

        out_dict = {'prediction': prediction,
                    'check': check_list,
                    'u_vectors': cf_u_vectors}
        return out_dict

class BiasedMF_PCFR(BiasedMF):
    def _init_nn(self):
        BiasedMF._init_nn(self)
        self._init_sensitive_filter()

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict['X'][:, 0] - 1
        i_ids = feed_dict['X'][:, 1] - 1

        # bias
        u_bias = self.user_bias(u_ids).view([-1])
        i_bias = self.item_bias(i_ids).view([-1])

        cf_u_vectors = self.uid_embeddings(u_ids)
        cf_i_vectors = self.iid_embeddings(i_ids)

        cf_u_vectors = self.filter(cf_u_vectors)

        prediction = (cf_u_vectors * cf_i_vectors).sum(dim=1).view([-1])
        prediction = prediction + u_bias + i_bias + self.global_bias

        out_dict = {'prediction': prediction,
                    'check': check_list,
                    'u_vectors': cf_u_vectors}
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


class BiasedMF_FairRec(BiasedMF):
    def _init_nn(self):
        BiasedMF._init_nn(self)
        self._init_sensitive_learner()
        self._init_sensitive_filter()
    

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict['X'][:, 0] - 1
        i_ids = feed_dict['X'][:, 1] - 1

        # bias
        u_bias = self.user_bias(u_ids).view([-1])
        i_bias = self.item_bias(i_ids).view([-1])

        cf_u_vectors = self.uid_embeddings(u_ids)
        cf_i_vectors = self.iid_embeddings(i_ids)

        cf_u_learner_vectors = self.learner(cf_u_vectors)
        cf_u_filter_vectors = self.filter(cf_u_vectors)

        if self.training:
            cf_u_vectors = cf_u_learner_vectors + cf_u_filter_vectors            
        else:
            cf_u_vectors = cf_u_filter_vectors

        prediction = (cf_u_vectors * cf_i_vectors).sum(dim=1).view([-1])
        prediction = prediction + u_bias + i_bias + self.global_bias

        out_dict = {'prediction': prediction,
                    'check': check_list,
                    'u_vectors': cf_u_vectors,
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

class BiasedMF_FOCF_AbsUnf(BiasedMF):
    def forward(self, feed_dict):
        out_dict = self.predict(feed_dict)
        batch_size = int(feed_dict[LABEL].shape[0] / 2)
        pos, neg = out_dict['prediction'][:batch_size], out_dict['prediction'][batch_size:]
        loss = -(pos - neg).sigmoid().log().sum()
        out_dict['loss'] = loss
        abs_loss = torch.abs(out_dict['prediction'] - feed_dict[LABEL])
        out_dict['abs_loss'] = abs_loss
        return out_dict

class BiasedMF_FOCF_ValUnf(BiasedMF):
    def forward(self, feed_dict):
        out_dict = self.predict(feed_dict)
        batch_size = int(feed_dict[LABEL].shape[0] / 2)
        pos, neg = out_dict['prediction'][:batch_size], out_dict['prediction'][batch_size:]
        loss = -(pos - neg).sigmoid().log().sum()
        out_dict['loss'] = loss
        val_loss = out_dict['prediction'] - feed_dict[LABEL]
        out_dict['val_loss'] = val_loss
        return out_dict
        
