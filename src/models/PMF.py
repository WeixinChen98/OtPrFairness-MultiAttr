import numpy as np
import torch
from torch import nn
from models.BaseRecModel import BaseRecModel
from utils.constants import *



class PMF(BaseRecModel):
    def _init_nn(self):
        self.uid_embeddings = nn.Embedding(self.user_num, self.u_vector_size)
        self.iid_embeddings = nn.Embedding(self.item_num, self.u_vector_size)

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict['X'][:, 0] - 1
        i_ids = feed_dict['X'][:, 1] - 1

        pmf_u_vectors = self.uid_embeddings(u_ids)
        pmf_i_vectors = self.iid_embeddings(i_ids)

        prediction = (pmf_u_vectors * pmf_i_vectors).sum(dim=1).view([-1])

        out_dict = {'prediction': prediction,
                    'check': check_list,
                    'u_vectors': pmf_u_vectors}
                    
        return out_dict


# filter_mode = 'separate' on single or all feature(s), similar to PFRec
# Hence remove the personalized setting due to the same fairness requirement for all users
class PMF_PCFR(PMF):
    def _init_nn(self):
        PMF._init_nn(self)
        self._init_sensitive_filter()

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict['X'][:, 0] - 1
        i_ids = feed_dict['X'][:, 1] - 1

        pmf_u_vectors = self.uid_embeddings(u_ids)
        pmf_i_vectors = self.iid_embeddings(i_ids)

        pmf_u_vectors = self.filter(pmf_u_vectors)

        prediction = (pmf_u_vectors * pmf_i_vectors).sum(dim=1).view([-1])

        out_dict = {'prediction': prediction,
                    'check': check_list,
                    'u_vectors': pmf_u_vectors}
                    
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


class PMF_FairRec(PMF):
    def _init_nn(self):
        PMF._init_nn(self)
        self._init_sensitive_learner()
        self._init_sensitive_filter()

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict['X'][:, 0] - 1
        i_ids = feed_dict['X'][:, 1] - 1

        pmf_u_vectors = self.uid_embeddings(u_ids)
        pmf_i_vectors = self.iid_embeddings(i_ids)

        pmf_u_learner_vectors = self.learner(pmf_u_vectors)
        pmf_u_filter_vectors = self.filter(pmf_u_vectors)

        if self.training:
            pmf_u_vectors = pmf_u_learner_vectors + pmf_u_filter_vectors            
        else:
            pmf_u_vectors = pmf_u_filter_vectors

        prediction = (pmf_u_vectors * pmf_i_vectors).sum(dim=1).view([-1])

        out_dict = {'prediction': prediction,
                    'check': check_list,
                    'u_vectors': pmf_u_vectors,
                    'u_learner_vectors': pmf_u_learner_vectors,
                    'u_filter_vectors': pmf_u_filter_vectors}
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

class PMF_FOCF_AbsUnf(PMF):
    def forward(self, feed_dict):
        out_dict = self.predict(feed_dict)
        batch_size = int(feed_dict[LABEL].shape[0] / 2)
        pos, neg = out_dict['prediction'][:batch_size], out_dict['prediction'][batch_size:]
        loss = -(pos - neg).sigmoid().log().sum()
        out_dict['loss'] = loss
        abs_loss = torch.abs(out_dict['prediction'] - feed_dict[LABEL])
        out_dict['abs_loss'] = abs_loss
        return out_dict

class PMF_FOCF_ValUnf(PMF):
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
class PMF_SolutionA(PMF):
    def _init_nn(self):
        PMF._init_nn(self)
        self._init_sensitive_filter()

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict['X'][:, 0] - 1
        i_ids = feed_dict['X'][:, 1] - 1

        pmf_u_vectors = self.uid_embeddings(u_ids)
        pmf_i_vectors = self.iid_embeddings(i_ids)

        pmf_u_vectors = self.filter(pmf_u_vectors)

        prediction = (pmf_u_vectors * pmf_i_vectors).sum(dim=1).view([-1])

        out_dict = {'prediction': prediction,
                    'check': check_list,
                    'u_vectors': pmf_u_vectors}
                    
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

    