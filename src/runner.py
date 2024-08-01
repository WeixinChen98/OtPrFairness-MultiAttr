# coding=utf-8
from utils.metrics import *
from sklearn.metrics import *
from sklearn.preprocessing import LabelBinarizer
from utils.generic import *
from utils.constants import *
import itertools as it
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity, smooth_l1_loss
from time import time
import numpy as np
import pandas as pd
import gc
import os


class RecRunner:
    @staticmethod
    def parse_runner_args(parser):
        """
        跑模型的命令行参数
        :param parser:
        :return:
        """
        parser.add_argument('--load', type=int, default=0,
                            help='Whether load model and continue to train')
        parser.add_argument('--load_attack', action='store_true',
                            help='Whether load attacker model and continue to train')
        parser.add_argument('--epoch', type=int, default=100,
                            help='Number of epochs.')
        parser.add_argument('--disc_epoch', type=int, default=500,
                            help='Number of epochs for training extra discriminator.')
        parser.add_argument('--check_epoch', type=int, default=1,
                            help='Check every epochs.')
        parser.add_argument('--early_stop', type=int, default=1,
                            help='whether to early-stop.')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='Learning rate.')
        parser.add_argument('--lr_attack', type=float, default=0.001,
                            help='attacker learning rate.')
        parser.add_argument('--batch_size', type=int, default=128,
                            help='Batch size during training.')
        parser.add_argument('--vt_batch_size', type=int, default=512,
                            help='Batch size during testing.')
        parser.add_argument('--dropout', type=float, default=0.2,
                            help='Dropout probability for each deep layer')
        parser.add_argument('--l2', type=float, default=1e-4,
                            help='Weight of l2_regularize in loss.')
        parser.add_argument('--l2_attack', type=float, default=1e-4,
                            help='Weight of attacker l2_regularize in loss.')
        parser.add_argument('--optimizer', type=str, default='GD',
                            help='optimizer: GD, Adam, Adagrad')
        parser.add_argument('--metric', type=str, default="RMSE",
                            help='metrics: RMSE, MAE, AUC, F1, Accuracy, Precision, Recall')
        parser.add_argument('--disc_metric', type=str, default="auc,acc,f1_macro,f1_micro",
                            help='metrics: auc, acc, f1_macro, f1_micro')
        parser.add_argument('--skip_eval', type=int, default=0,
                            help='number of epochs without evaluation')
        parser.add_argument('--num_worker', type=int, default=0,
                            help='number of processes for multi-processing data loading.')
        parser.add_argument('--eval_disc', action='store_true',
                            help='train extra discriminator for evaluation.')
        parser.add_argument('--fairrec_lambda', type=float, default=0.5,
                            help='fairrec_lambda.')
        return parser

    def __init__(self, optimizer='GD', learning_rate=0.01, epoch=100, batch_size=128, eval_batch_size=128 * 128,
                 dropout=0.2, l2=1e-5, metrics='RMSE', disc_metrics='auc,acc,f1_macro,f1_micro', check_epoch=10, early_stop=1, num_worker=0,
                disc_epoch=1000):
        """
        初始化
        :param optimizer: optimizer name
        :param learning_rate: learning rate
        :param epoch: total training epochs
        :param batch_size: batch size for training
        :param eval_batch_size: batch size for evaluation
        :param dropout: dropout rate
        :param l2: l2 weight
        :param metrics: evaluation metrics list
        :param check_epoch: check intermediate results in every n epochs
        :param early_stop: 1 for early stop, 0 for not.
        :param disc_epoch: number of epoch for training extra discriminator
        """
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.dropout = dropout
        self.no_dropout = 0.0
        self.l2_weight = l2
        self.disc_epoch = disc_epoch

        # convert metrics to list of str
        self.metrics = metrics.lower().split(',')
        self.disc_metrics = disc_metrics.lower().split(',')
        self.check_epoch = check_epoch
        self.early_stop = early_stop
        self.time = None

        # record train, validation, test results
        self.train_results, self.valid_results = [], []
        self.pred_results, self.disc_results = [], []

        self.num_worker = num_worker

    def _build_optimizer(self, model, lr=None, l2_weight=None):
        optimizer_name = self.optimizer_name.lower()
        if lr is None:
            lr = self.learning_rate
        if l2_weight is None:
            l2_weight = self.l2_weight

        if optimizer_name == 'gd':
            logging.info("Optimizer: GD")
            optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, weight_decay=l2_weight)
        elif optimizer_name == 'adagrad':
            logging.info("Optimizer: Adagrad")
            optimizer = torch.optim.Adagrad(
                model.parameters(), lr=lr, weight_decay=l2_weight)
        elif optimizer_name == 'adam':
            logging.info("Optimizer: Adam")
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=l2_weight)
        else:
            logging.error("Unknown Optimizer: " + self.optimizer_name)
            assert self.optimizer_name in ['GD', 'Adagrad', 'Adam']
            optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, weight_decay=l2_weight)
        return optimizer

    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    @staticmethod
    def _get_masked_disc(disc_dict, labels, mask):
        if np.sum(mask) == 0:
            return []
        masked_disc_label = [(disc_dict[i + 1], labels[:, i])
                             for i, val in enumerate(mask) if val != 0]
        return masked_disc_label

    def fit(self, model, batches, epoch=-1):  # fit the results for an input set
        """
        Train the model
        :param model: model instance
        :param batches: train data in batches
        :param epoch: epoch number
        :return: return the output of the last round
        """
        gc.collect()
        torch.cuda.empty_cache()

        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        model.train()

        loss_list = list()
        output_dict = dict()
        for batch in tqdm(batches, leave=False, desc='Epoch %5d' % (epoch + 1),
                          ncols=100, mininterval=1):

            batch = batch_to_gpu(batch)
            model.optimizer.zero_grad()

            result_dict = model(batch)
            rec_loss = result_dict['loss']

            loss = rec_loss
            loss.backward()
            model.optimizer.step()

            loss_list.append(result_dict['loss'].detach().cpu().data.numpy())
            output_dict['check'] = result_dict['check']

        output_dict['loss'] = np.mean(loss_list)
        return output_dict

    def train(self, model, dp_dict, skip_eval=0):
        """
        Train model
        :param model: model obj
        :param dp_dict: Data processors for train valid and test
        :param skip_eval: number of epochs to skip for evaluations
        :return:
        """
        train_data = DataLoader(dp_dict['train'], batch_size=self.batch_size, num_workers=self.num_worker,
                                shuffle=True, collate_fn=dp_dict['train'].collate_fn)
        validation_data = DataLoader(dp_dict['valid'], batch_size=None, num_workers=self.num_worker,
                                     pin_memory=True, collate_fn=dp_dict['valid'].collate_fn)

        self._check_time(start=True)  # start time
        try:
            for epoch in range(self.epoch):
                self._check_time()
                output_dict = \
                    self.fit(model, train_data, epoch=epoch)
                if self.check_epoch > 0 and (epoch == 1 or epoch % self.check_epoch == 0):
                    self.check(model, output_dict)
                training_time = self._check_time()

                if epoch >= skip_eval:
                    valid_result = self.evaluate(model, validation_data) if \
                        validation_data is not None else [-1.0] * len(self.metrics)
                    testing_time = self._check_time()

                    # self.train_results.append(train_result)
                    self.valid_results.append(valid_result)

                    logging.info("Epoch %5d [%.1f s]\n validation= %s [%.1f s] "
                                 % (epoch + 1, training_time,
                                    format_metric(valid_result),
                                    testing_time) + ','.join(self.metrics))

                    if best_result(self.metrics[0], self.valid_results) == self.valid_results[-1]:
                        model.save_model()

                    if self.eva_termination() and self.early_stop == 1:
                        logging.info(
                            "Early stop at %d based on validation result." % (epoch + 1))
                        break
                if epoch < skip_eval:
                    logging.info("Epoch %5d [%.1f s]" %
                                 (epoch + 1, training_time))

        except KeyboardInterrupt:
            logging.info("Early stop manually")
            save_here = input("Save here? (1/0) (default 0):")
            if str(save_here).lower().startswith('1'):
                model.save_model()

        # Find the best validation result across iterations
        best_valid_score = best_result(self.metrics[0], self.valid_results)
        best_epoch = self.valid_results.index(best_valid_score)
        logging.info("Best Iter (validation) = %5d\t valid= %s [%.1f s] "
                     % (best_epoch + 1,
                        format_metric(self.valid_results[best_epoch]),
                        self.time[1] - self.time[0]) + ','.join(self.metrics))

        model.load_model()

    @torch.no_grad()
    def evaluate(self, model, batches, metrics=None):
        """
        evaluate recommendation performance
        :param model:
        :param batches: data batches, each batch is a dict.
        :param mask: filter mask
        :param metrics: list of str
        :return: list of float number for each metric
        """
        if metrics is None:
            metrics = self.metrics
        model.eval()

        prediction = []
        data_dict = {LABEL: [], USER: []}
        for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Predict'):
            batch = batch_to_gpu(batch)
            out_dict = model.predict(batch)
            batch_prediction = out_dict['prediction']
            batch_labels = batch[LABEL].cpu()
            batch_user_ids = batch['X'][:, 0].cpu()
            assert len(batch_labels) == len(batch_prediction)
            assert len(batch_user_ids == len(batch_prediction))
            batch_prediction = batch_prediction.cpu().numpy()
            prediction.extend(batch_prediction)
            data_dict[LABEL].extend(batch_labels)
            data_dict[USER].extend(batch_user_ids)


        results = self.evaluate_method(prediction, data_dict, metrics=metrics)
        
        evaluations = []
        for metric in metrics:
            evaluations.append(np.average(results[metric]))

        return evaluations

    @torch.no_grad()
    def generate_rank_file(self, model, batches, metrics=None):
        """
        evaluate recommendation performance
        :param model:
        :param batches: data batches, each batch is a dict.
        :param mask: filter mask
        :param metrics: list of str
        :return: list of float number for each metric
        """
        if metrics is None:
            metrics = self.metrics
        model.eval()

        data_dict = pd.DataFrame({USER: [], ITEM: [], 'score': [], LABEL: []})
        for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Predict'):
            batch = batch_to_gpu(batch)
            out_dict = model.predict(batch)
            prediction = out_dict['prediction']
            labels = batch[LABEL].cpu()
            user_ids = batch['X'][:, 0].cpu()
            item_ids = batch['X'][:, 1].cpu()
            assert len(labels) == len(prediction)
            assert len(user_ids == len(prediction))
            prediction = prediction.cpu().numpy()
            batch_data_dict = pd.DataFrame({USER: user_ids, ITEM: item_ids, 'score': prediction, LABEL: labels})
            data_dict = pd.concat([data_dict, batch_data_dict], ignore_index=True)
        data_dict = data_dict.astype({USER:int, ITEM:int})
        return data_dict

    @staticmethod
    def evaluate_method(p, data, metrics):
        """
        Evaluate model predictions.
        :param p: predicted values, np.array
        :param data: data dictionary which include ground truth labels
        :param metrics: metrics list
        :return: a list of results. The order is consistent to metric list.
        """
        label = data[LABEL]
        evaluations = {}
        for metric in metrics:
            if metric == 'rmse':
                evaluations[metric] = [np.sqrt(mean_squared_error(label, p))]
            elif metric == 'mae':
                evaluations[metric] = [mean_absolute_error(label, p)]
            elif metric == 'auc':
                evaluations[metric] = [roc_auc_score(label, p)]
            else:
                k = int(metric.split('@')[-1])
                df = pd.DataFrame()
                df[USER] = [uid.item() for uid in data[USER]]
                df['p'] = p
                df['l'] = [l.item() for l in label]
                df = df.sort_values(by='p', ascending=False)
                df_group = df.groupby(USER)
                if metric.startswith('ndcg@'):
                    ndcgs = []
                    for uid, group in df_group:
                        ndcgs.append(ndcg_at_k(group['l'].tolist(), k=k, method=1))
                    evaluations[metric] = ndcgs
                elif metric.startswith('hit@'):
                    hits = []
                    for uid, group in df_group:
                        hits.append(int(np.sum(group['l'][:k]) > 0))
                    evaluations[metric] = hits
                elif metric.startswith('precision@'):
                    precisions = []
                    for uid, group in df_group:
                        precisions.append(precision_at_k(group['l'].tolist(), k=k))
                    evaluations[metric] = precisions
                elif metric.startswith('recall@'):
                    recalls = []
                    for uid, group in df_group:
                        recalls.append(1.0 * np.sum(group['l'][:k]) / np.sum(group['l']))
                    evaluations[metric] = recalls
                elif metric.startswith('f1@'):
                    f1 = []
                    for uid, group in df_group:
                        num_overlap = 1.0 * np.sum(group['l'][:k])
                        f1.append(2 * num_overlap /
                                  (k + 1.0 * np.sum(group['l'])))
                    evaluations[metric] = f1
        return evaluations

    def eva_termination(self):
        """
        Early stopper
        :return:
        """
        metric = self.metrics[0]
        valid = self.valid_results
        if len(valid) > 20 and metric in LOWER_METRIC_LIST and strictly_increasing(valid[-5:]):
            return True
        elif len(valid) > 20 and metric not in LOWER_METRIC_LIST and strictly_decreasing(valid[-5:]):
            return True
        elif len(valid) - valid.index(best_result(metric, valid)) > 20:
            return True
        return False

    @torch.no_grad()
    def _eval_discriminator(self, model, labels, u_vectors, fair_disc_dict, num_disc):
        feature_info = model.data_processor_dict['train'].data_reader.feature_info
        feature_eval_dict = {}
        for i in range(num_disc):
            discriminator = fair_disc_dict[i + 1]
            label = labels[:, i]
            feature_name = feature_info[i + 1].name
            discriminator.eval()
            if feature_info[i + 1].num_class == 2:
                prediction = discriminator.predict(
                    u_vectors)['prediction'].squeeze()
            else:
                prediction = discriminator.predict(u_vectors)['output']
            feature_eval_dict[feature_name] = {'label': label.cpu(), 'prediction': prediction.detach().cpu(),
                                               'num_class': feature_info[i + 1].num_class}
            discriminator.train()
        return feature_eval_dict

    @staticmethod
    def _disc_eval_method(label, prediction, num_class, metrics=['auc', 'acc', 'f1_macro', 'f1_micro']):
        res = []
        for metric in metrics:
            if metric == 'auc':
                if num_class == 2:
                    score = roc_auc_score(label, prediction, average='micro')
                    # score = roc_auc_score(label, prediction)
                    score = max(score, 1 - score)
                else:
                    lb = LabelBinarizer()
                    classes = [i for i in range(num_class)]
                    lb.fit(classes)
                    label = lb.transform(label)
                    # label = lb.fit_transform(label)
                    score = roc_auc_score(
                        label, prediction, multi_class='ovo', average='macro')
                    score = max(score, 1 - score)
            elif metric == 'acc':
                score = accuracy_score(label, prediction)
            elif metric == 'f1_macro':
                score = f1_score(label, prediction, average='macro')
            elif metric == 'f1_micro':
                score = f1_score(label, prediction, average='micro')
            elif metric == 'f1_binary':
                score = f1_score(label, prediction, average='binary')
            else:
                raise ValueError(
                    'Unknown evaluation metric in _disc_eval_method().')
            res.append(score)
        return res

    def check(self, model, out_dict):
        """
        Check intermediate results
        :param model: model obj
        :param out_dict: output dictionary
        :return:
        """
        check = out_dict
        logging.info(os.linesep)
        for i, t in enumerate(check['check']):
            d = np.array(t[1].detach().cpu())
            logging.info(os.linesep.join(
                [t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]) + os.linesep)

        loss, l2 = check['loss'], model.l2()
        l2 = l2 * self.l2_weight
        l2 = l2.detach()
        logging.info('loss = %.4f, l2 = %.4f' % (loss, l2))

    def train_discriminator(self, model, dp_dict, fair_disc_dict, lr_attack=None, l2_attack=None):
        """
        Train discriminator to evaluate the quality of learned embeddings
        :param model: trained model
        :param dp_dict: Data processors for train valid and test
        :param fair_disc_dict: fairness discriminator dictionary
        :return:
        """
        train_data = DataLoader(dp_dict['train'], batch_size=dp_dict['train'].batch_size, num_workers=self.num_worker,
                                shuffle=True, collate_fn=dp_dict['train'].collate_fn)
        test_data = DataLoader(dp_dict['test'], batch_size=dp_dict['test'].batch_size, num_workers=self.num_worker,
                               pin_memory=True, collate_fn=dp_dict['test'].collate_fn)
        self._check_time(start=True)  # 记录初始时间s

        feature_results = defaultdict(list)
        best_results = dict()
        try:
            for epoch in range(self.disc_epoch):
                self._check_time()
                output_dict = \
                    self.fit_disc(model, train_data, fair_disc_dict, epoch=epoch,
                                  lr_attack=lr_attack, l2_attack=l2_attack)

                if self.check_epoch > 0 and (epoch == 1 or epoch % (self.disc_epoch // 4) == 0):
                    self.check_disc(output_dict)
                training_time = self._check_time()

                test_result_dict = \
                    self.evaluation_disc(
                        model, fair_disc_dict, test_data, dp_dict['train'])
                d_score_dict = test_result_dict['d_score']
                # testing_time = self._check_time()
                if epoch % (self.disc_epoch // 4) == 0:
                    logging.info("Epoch %5d [%.1f s]" %
                                 (epoch + 1, training_time))

                for f_name in d_score_dict:
                    if epoch % (self.disc_epoch // 4) == 0:
                        logging.info("%s disc test = %s" % (f_name, format_metric(
                            d_score_dict[f_name])) + ' ' + ', '.join(self.disc_metrics))
                    feature_results[f_name].append(d_score_dict[f_name][0])
                    if d_score_dict[f_name][0] == max(feature_results[f_name]):
                        best_results[f_name] = d_score_dict[f_name]
                        idx = dp_dict['train'].data_reader.f_name_2_idx[f_name]
                        fair_disc_dict[idx].save_model()

        except KeyboardInterrupt:
            logging.info("Early stop manually")
            save_here = input("Save here? (1/0) (default 0):")
            if str(save_here).lower().startswith('1'):
                for idx in fair_disc_dict:
                    fair_disc_dict[idx].save_model()

        for f_name in best_results:
            logging.info("{} best {}: {:.4f}".format(
                f_name, self.disc_metrics[0], best_results[f_name][0]))
            logging.info("And the corresponding %s best disc test= %s" % (
                f_name, format_metric(best_results[f_name])) + ' ' + ', '.join(self.disc_metrics))

        for idx in fair_disc_dict:
            fair_disc_dict[idx].load_model()

    def fit_disc(self, model, batches, fair_disc_dict, epoch=-1, lr_attack=None, l2_attack=None):
        """
        Train the discriminator
        :param model: model instance
        :param batches: train data in batches
        :param fair_disc_dict: fairness discriminator dictionary
        :param epoch: epoch number
        :param lr_attack: attacker learning rate
        :param l2_attack: l2 regularization weight for attacker
        :return: return the output of the last round
        """
        gc.collect()
        torch.cuda.empty_cache()

        for idx in fair_disc_dict:
            discriminator = fair_disc_dict[idx]
            if discriminator.optimizer is None:
                discriminator.optimizer = self._build_optimizer(
                    discriminator, lr=lr_attack, l2_weight=l2_attack)
            discriminator.train()

        output_dict = dict()
        loss_acc = defaultdict(list)

        for batch in tqdm(batches, leave=False, desc='Epoch %5d' % (epoch + 1),
                          ncols=100, mininterval=1):
            mask = [1] * model.num_features
            mask = np.asarray(mask)

            batch = batch_to_gpu(batch)

            labels = batch['features']
            masked_disc_label = \
                self._get_masked_disc(fair_disc_dict, labels, mask)

            # calculate recommendation loss + fair discriminator penalty
            uids = batch['X'] - 1
            vectors = model.uid_embeddings(uids)
            output_dict['check'] = []

            # update discriminator
            if len(masked_disc_label) != 0:
                for idx, (discriminator, label) in enumerate(masked_disc_label):
                    discriminator.optimizer.zero_grad()
                    disc_loss = discriminator(vectors.detach(), label)
                    disc_loss.backward()
                    discriminator.optimizer.step()
                    loss_acc[discriminator.name].append(
                        disc_loss.detach().cpu())

        for key in loss_acc:
            loss_acc[key] = np.mean(loss_acc[key])

        output_dict['loss'] = loss_acc
        return output_dict

    @torch.no_grad()
    def evaluation_disc(self, model, fair_disc_dict, test_data, dp):
        num_features = dp.data_reader.num_features

        def eval_disc(labels, u_vectors, fair_disc_dict, mask):
            feature_info = dp.data_reader.feature_info
            feature_eval_dict = {}
            for i, val in enumerate(mask):
                if val == 0:
                    continue
                discriminator = fair_disc_dict[i + 1]
                label = labels[:, i]
                feature_name = feature_info[i + 1].name
                discriminator.eval()
                if feature_info[i + 1].num_class == 2:
                    prediction = discriminator.predict(
                        u_vectors)['prediction'].squeeze()
                else:
                    prediction = discriminator.predict(u_vectors)['output']
                feature_eval_dict[feature_name] = {'label': label.cpu(), 'prediction': prediction.detach().cpu(),
                                                   'num_class': feature_info[i + 1].num_class}
                discriminator.train()
            return feature_eval_dict

        eval_dict = {}
        for batch in test_data:
            mask = [1] * num_features

            batch = batch_to_gpu(batch)

            labels = batch['features']
            uids = batch['X'] - 1
            vectors = model.uid_embeddings(uids)

            batch_eval_dict = eval_disc(
                labels, vectors.detach(), fair_disc_dict, mask)
            for f_name in batch_eval_dict:
                if f_name not in eval_dict:
                    eval_dict[f_name] = batch_eval_dict[f_name]
                else:
                    new_label = batch_eval_dict[f_name]['label']
                    current_label = eval_dict[f_name]['label']
                    eval_dict[f_name]['label'] = torch.cat(
                        (current_label, new_label), dim=0)

                    new_prediction = batch_eval_dict[f_name]['prediction']
                    current_prediction = eval_dict[f_name]['prediction']
                    eval_dict[f_name]['prediction'] = torch.cat(
                        (current_prediction, new_prediction), dim=0)

        # generate discriminator evaluation scores
        d_score_dict = {}
        if eval_dict is not None:
            for f_name in eval_dict:
                l = eval_dict[f_name]['label']
                pred = eval_dict[f_name]['prediction']
                n_class = eval_dict[f_name]['num_class']
                d_score_dict[f_name] = self._disc_eval_method(
                    l, pred, n_class, metrics=self.disc_metrics)

        output_dict = dict()
        output_dict['d_score'] = d_score_dict
        return output_dict



    @staticmethod
    def check_disc(out_dict):
        check = out_dict
        logging.info(os.linesep)
        for i, t in enumerate(check['check']):
            d = np.array(t[1].detach().cpu())
            logging.info(os.linesep.join(
                [t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]) + os.linesep)

        loss_dict = check['loss']
        for disc_name, disc_loss in loss_dict.items():
            logging.info('%s loss = %.4f' % (disc_name, disc_loss))

        # for discriminator
        if 'd_score' in out_dict:
            disc_score_dict = out_dict['d_score']
            for feature in disc_score_dict:
                logging.info('{} AUC = {:.4f}'.format(
                    feature, disc_score_dict[feature]))


class RecRunner_PCFR(RecRunner):
    def fit(self, model, batches, fair_disc_dict, epoch=-1):  # fit the results for an input set
        """
        Train the model
        :param model: model instance
        :param batches: train data in batches
        :param fair_disc_dict: fairness discriminator dictionary
        :param epoch: epoch number
        :return: return the output of the last round
        """
        gc.collect()
        torch.cuda.empty_cache()

        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        model.train()

        for idx in fair_disc_dict:
            discriminator = fair_disc_dict[idx]
            if discriminator.optimizer is None:
                discriminator.optimizer = self._build_optimizer(discriminator)
            discriminator.train()

        loss_list = list()
        output_dict = dict()
        eval_dict = None
        for batch in tqdm(batches, leave=False, desc='Epoch %5d' % (epoch + 1),
                          ncols=100, mininterval=1):

            batch = batch_to_gpu(batch)
            model.optimizer.zero_grad()

            # for all features considered
            mask = [1] * model.num_features
            mask = np.asarray(mask)

            labels = batch['features'][:len(batch['features'])//2, :]
            masked_disc_label = self._get_masked_disc(
                fair_disc_dict, labels, mask)

            result_dict = model(batch)
            rec_loss = result_dict['loss']
            vectors = result_dict['u_vectors']
            vectors = vectors[:len(vectors) // 2, :]

            fair_d_penalty = 0
            for fair_disc, label in masked_disc_label:
                fair_d_penalty += fair_disc(vectors, label)
            fair_d_penalty *= -1
            loss = rec_loss + fair_d_penalty

            loss.backward()
            model.optimizer.step()

            loss_list.append(result_dict['loss'].detach().cpu().data.numpy())
            output_dict['check'] = result_dict['check']
            if len(masked_disc_label) != 0:
                for discriminator, label in masked_disc_label:
                    discriminator.optimizer.zero_grad()
                    disc_loss = discriminator(vectors.detach(), label)
                    disc_loss.backward(retain_graph=False)
                    discriminator.optimizer.step()

            # collect discriminator evaluation results
            if eval_dict is None:
                eval_dict = self._eval_discriminator(
                    model, labels, vectors.detach(), fair_disc_dict, len(mask))
            else:
                batch_eval_dict = self._eval_discriminator(
                    model, labels, vectors.detach(), fair_disc_dict, len(mask))
                for f_name in eval_dict:
                    new_label = batch_eval_dict[f_name]['label']
                    current_label = eval_dict[f_name]['label']
                    eval_dict[f_name]['label'] = torch.cat(
                        (current_label, new_label), dim=0)

                    new_prediction = batch_eval_dict[f_name]['prediction']
                    current_prediction = eval_dict[f_name]['prediction']
                    eval_dict[f_name]['prediction'] = torch.cat(
                        (current_prediction, new_prediction), dim=0)

        # generate discriminator evaluation scores
        d_score_dict = {}
        if eval_dict is not None:
            for f_name in eval_dict:
                l = eval_dict[f_name]['label']
                pred = eval_dict[f_name]['prediction']
                n_class = eval_dict[f_name]['num_class']
                d_score_dict[f_name] = self._disc_eval_method(l, pred, n_class)

        output_dict['d_score'] = d_score_dict
        output_dict['loss'] = np.mean(loss_list)
        return output_dict

    def train(self, model, dp_dict, fair_disc_dict, skip_eval=0):
        """
        Train model
        :param model: model obj
        :param dp_dict: Data processors for train valid and test
        :param skip_eval: number of epochs to skip for evaluations
        :param fair_disc_dict: fairness discriminator dictionary
        :return:
        """
        train_data = DataLoader(dp_dict['train'], batch_size=self.batch_size, num_workers=self.num_worker,
                                shuffle=True, collate_fn=dp_dict['train'].collate_fn)
        validation_data = DataLoader(dp_dict['valid'], batch_size=None, num_workers=self.num_worker,
                                     pin_memory=True, collate_fn=dp_dict['test'].collate_fn)

        self._check_time(start=True)  # start time
        try:
            for epoch in range(self.epoch):
                self._check_time()
                output_dict = \
                    self.fit(model, train_data, fair_disc_dict, epoch=epoch)
                if self.check_epoch > 0 and (epoch == 1 or epoch % self.check_epoch == 0):
                    self.check(model, output_dict)
                training_time = self._check_time()

                if epoch >= skip_eval:
                    valid_result = self.evaluate(model, validation_data) if \
                        validation_data is not None else [-1.0] * len(self.metrics)
                    testing_time = self._check_time()

                    # self.train_results.append(train_result)
                    self.valid_results.append(valid_result)
                    self.disc_results.append(output_dict['d_score'])

                    logging.info("Epoch %5d [%.1f s]\n validation= %s [%.1f s] "
                                 % (epoch + 1, training_time,
                                    format_metric(valid_result),
                                    testing_time) + ','.join(self.metrics))

                    if best_result(self.metrics[0], self.valid_results) == self.valid_results[-1]:
                        model.save_model()
                        for idx in fair_disc_dict:
                            fair_disc_dict[idx].save_model()

                    if self.eva_termination() and self.early_stop == 1:
                        logging.info(
                            "Early stop at %d based on validation result." % (epoch + 1))
                        break
                if epoch < skip_eval:
                    logging.info("Epoch %5d [%.1f s]" %
                                 (epoch + 1, training_time))

        except KeyboardInterrupt:
            logging.info("Early stop manually")
            save_here = input("Save here? (1/0) (default 0):")
            if str(save_here).lower().startswith('1'):
                model.save_model()
                for idx in fair_disc_dict:
                    fair_disc_dict[idx].save_model()

        # Find the best validation result across iterations
        best_valid_score = best_result(self.metrics[0], self.valid_results)
        best_epoch = self.valid_results.index(best_valid_score)

        # prepare disc result string
        disc_info = self.disc_results[best_epoch]
        disc_info_str = ['{}={:.4f}'.format(
            key, disc_info[key][0]) for key in disc_info]
        disc_info_str = ','.join(disc_info_str)

        logging.info("Best Iter (validation) = %5d\t valid= %s [%.1f s] "
                     % (best_epoch + 1,
                        format_metric(self.valid_results[best_epoch]),
                        self.time[1] - self.time[0]) + ','.join(self.metrics) + ' ' + disc_info_str + self.disc_metrics[0])

        model.load_model()
        for idx in fair_disc_dict:
            fair_disc_dict[idx].load_model()

    def check(self, model, out_dict):
        """
        Check intermediate results
        :param model: model obj
        :param out_dict: output dictionary
        :return:
        """
        check = out_dict
        logging.info(os.linesep)
        for i, t in enumerate(check['check']):
            d = np.array(t[1].detach().cpu())
            logging.info(os.linesep.join(
                [t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]) + os.linesep)

        loss, l2 = check['loss'], model.l2()
        l2 = l2 * self.l2_weight
        l2 = l2.detach()
        logging.info('loss = %.4f, l2 = %.4f' % (loss, l2))

        # for discriminator
        disc_score_dict = out_dict['d_score']
        for feature in disc_score_dict:
            logging.info("%s disc test= %s" % (feature, format_metric(
                disc_score_dict[feature])) + ', '.join(self.disc_metrics))

    def fit_disc(self, model, batches, fair_disc_dict, epoch=-1, lr_attack=None, l2_attack=None):
        """
        Train the discriminator
        :param model: model instance
        :param batches: train data in batches
        :param fair_disc_dict: fairness discriminator dictionary
        :param epoch: epoch number
        :param lr_attack: attacker learning rate
        :param l2_attack: l2 regularization weight for attacker
        :return: return the output of the last round
        """
        gc.collect()
        torch.cuda.empty_cache()

        for idx in fair_disc_dict:
            discriminator = fair_disc_dict[idx]
            if discriminator.optimizer is None:
                discriminator.optimizer = self._build_optimizer(
                    discriminator, lr=lr_attack, l2_weight=l2_attack)
            discriminator.train()

        output_dict = dict()
        loss_acc = defaultdict(list)

        for batch in tqdm(batches, leave=False, desc='Epoch %5d' % (epoch + 1),
                          ncols=100, mininterval=1):
            mask = [1] * model.num_features
            mask = np.asarray(mask)

            batch = batch_to_gpu(batch)

            labels = batch['features']
            masked_disc_label = \
                self._get_masked_disc(fair_disc_dict, labels, mask)

            # calculate recommendation loss + fair discriminator penalty
            uids = batch['X'] - 1
            vectors = model.filter(model.uid_embeddings(uids))
            output_dict['check'] = []

            # update discriminator
            if len(masked_disc_label) != 0:
                for idx, (discriminator, label) in enumerate(masked_disc_label):
                    discriminator.optimizer.zero_grad()
                    disc_loss = discriminator(vectors.detach(), label)
                    disc_loss.backward()
                    discriminator.optimizer.step()
                    loss_acc[discriminator.name].append(
                        disc_loss.detach().cpu())

        for key in loss_acc:
            loss_acc[key] = np.mean(loss_acc[key])

        output_dict['loss'] = loss_acc
        return output_dict


class RecRunner_FOCF_AbsUnf(RecRunner):
    def __init__(self, group_0_ids, group_1_ids, optimizer='GD', learning_rate=0.01, epoch=100, batch_size=128, eval_batch_size=128 * 128,
                 dropout=0.2, l2=1e-5, metrics='RMSE', disc_metrics='auc,acc,f1_macro,f1_micro', check_epoch=10, early_stop=1, num_worker=0,
                 disc_epoch=1000):
        RecRunner.__init__(self, optimizer, learning_rate, epoch, batch_size, eval_batch_size, dropout, l2,
                      metrics, disc_metrics, check_epoch, early_stop, num_worker, disc_epoch)
        self.group_0_ids = group_0_ids
        self.group_1_ids = group_1_ids

    def fit(self, model, batches, epoch=-1):  # fit the results for an input set
        """
        Train the model
        :param model: model instance
        :param batches: train data in batches
        :param epoch: epoch number
        :return: return the output of the last round
        """
        gc.collect()
        torch.cuda.empty_cache()

        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        model.train()

        loss_list = list()
        output_dict = dict()
        for batch in tqdm(batches, leave=False, desc='Epoch %5d' % (epoch + 1),
                          ncols=100, mininterval=1):

            batch = batch_to_gpu(batch)
            model.optimizer.zero_grad()

            result_dict = model(batch)
            rec_loss = result_dict['loss']
            abs_loss = result_dict['abs_loss']

            # adding absolute unfairness loss to learning objectives
            uids = batch['X'][:, 0]

            group_0_u_mask = torch.Tensor([1 if uid.item() in self.group_0_ids else 0 for uid in uids]).cuda()
            group_1_u_mask = torch.Tensor([1 if uid.item() in self.group_1_ids else 0 for uid in uids]).cuda()
            group_0_abs_loss = abs_loss * group_0_u_mask
            group_1_abs_loss = abs_loss * group_1_u_mask

            eps = 1e-10
            fair_loss_input = torch.abs(torch.sum(group_0_abs_loss)/(torch.sum(group_0_u_mask) + eps) - torch.sum(group_1_abs_loss)/(torch.sum(group_1_u_mask) + eps))
            fair_loss_target = torch.zeros_like(fair_loss_input)

            # multiple batch_size for equal weighting 
            fair_loss = smooth_l1_loss(fair_loss_input, fair_loss_target) * (len(batch['X'])//2)
            loss = rec_loss + fair_loss

            loss.backward()
            model.optimizer.step()

            loss_list.append(result_dict['loss'].detach().cpu().data.numpy())
            output_dict['check'] = result_dict['check']

        output_dict['loss'] = np.mean(loss_list)
        return output_dict


class RecRunner_FOCF_ValUnf(RecRunner):
    def __init__(self, group_0_ids, group_1_ids, optimizer='GD', learning_rate=0.01, epoch=100, batch_size=128, eval_batch_size=128 * 128,
                 dropout=0.2, l2=1e-5, metrics='RMSE', disc_metrics='auc,acc,f1_macro,f1_micro', check_epoch=10, early_stop=1, num_worker=0,
                disc_epoch=1000):
        RecRunner.__init__(self, optimizer, learning_rate, epoch, batch_size, eval_batch_size, dropout, l2,
                      metrics, disc_metrics, check_epoch, early_stop, num_worker, disc_epoch)
        self.group_0_ids = group_0_ids
        self.group_1_ids = group_1_ids

    def fit(self, model, batches, epoch=-1):  # fit the results for an input set
        """
        Train the model
        :param model: model instance
        :param batches: train data in batches
        :param epoch: epoch number
        :return: return the output of the last round
        """
        gc.collect()
        torch.cuda.empty_cache()

        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        model.train()

        loss_list = list()
        output_dict = dict()
        for batch in tqdm(batches, leave=False, desc='Epoch %5d' % (epoch + 1),
                          ncols=100, mininterval=1):

            batch = batch_to_gpu(batch)
            model.optimizer.zero_grad()

            result_dict = model(batch)
            rec_loss = result_dict['loss']
            val_loss = result_dict['val_loss']

            # adding absolute unfairness loss to learning objectives
            uids = batch['X'][:, 0]

            group_0_u_mask = torch.Tensor([1 if uid.item() in self.group_0_ids else 0 for uid in uids]).cuda()
            group_1_u_mask = torch.Tensor([1 if uid.item() in self.group_1_ids else 0 for uid in uids]).cuda()
            group_0_val_loss = val_loss * group_0_u_mask
            group_1_val_loss = val_loss * group_1_u_mask

            eps = 1e-10
            fair_loss_input = torch.abs(torch.sum(group_0_val_loss)/(torch.sum(group_0_u_mask)+eps) - torch.sum(group_1_val_loss)/(torch.sum(group_1_u_mask)+eps))
            fair_loss_target = torch.zeros_like(fair_loss_input)

            # multipy batch_size for equal weighting 
            fair_loss = smooth_l1_loss(fair_loss_input, fair_loss_target) * (len(batch['X'])//2)
            loss = rec_loss + fair_loss

            loss.backward()
            model.optimizer.step()

            loss_list.append(result_dict['loss'].detach().cpu().data.numpy())
            output_dict['check'] = result_dict['check']

        output_dict['loss'] = np.mean(loss_list)
        return output_dict


class RecRunner_FairRec(RecRunner):
    def fit(self, model, batches, fair_pred_dict, fair_disc_dict, loss_lambda = 0.5, epoch=-1):
        gc.collect()
        torch.cuda.empty_cache()

        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        model.train()


        for idx in fair_pred_dict:
            predictor = fair_pred_dict[idx]
            if predictor.optimizer is None:
                predictor.optimizer = self._build_optimizer(predictor)
            predictor.train()

        for idx in fair_disc_dict:
            discriminator = fair_disc_dict[idx]
            if discriminator.optimizer is None:
                discriminator.optimizer = self._build_optimizer(discriminator)
            discriminator.train()
        
        loss_list = list()
        output_dict = dict()
        pred_eval_dict = None
        disc_eval_dict = None
        for batch in tqdm(batches, leave=False, desc='Epoch %5d' % (epoch + 1),
                          ncols=100, mininterval=1):
            mask = [1] * model.num_features
            mask = np.asarray(mask)

            batch = batch_to_gpu(batch)
            model.optimizer.zero_grad()

            labels = batch['features'][:len(batch['features'])//2, :]
            masked_pred_label = \
                self._get_masked_disc(fair_pred_dict, labels, mask)
            masked_disc_label = \
                self._get_masked_disc(fair_disc_dict, labels, mask)
                

            # calculate recommendation loss + fair discriminator penalty
            result_dict = model(batch)
            rec_loss = result_dict['loss']
            u_learner_vectors = result_dict['u_learner_vectors']
            u_filter_vectors = result_dict['u_filter_vectors']
            u_learner_vectors = u_learner_vectors[:len(u_learner_vectors) // 2, :]
            u_filter_vectors = u_filter_vectors[:len(u_filter_vectors) // 2, :]

            fair_pred_loss = 0
            fair_disc_loss = 0
            orth_reg_Loss = 0
            for fair_pred, label in masked_pred_label:
                fair_pred_loss += fair_pred(u_learner_vectors, label)

            for fair_disc, label in masked_disc_label:
                fair_disc_loss += fair_disc(u_filter_vectors, label)
            fair_disc_loss *= -1

            orth_reg_Loss = cosine_similarity(u_learner_vectors, u_filter_vectors).abs().mean()
            loss = rec_loss + loss_lambda * fair_disc_loss + loss_lambda * fair_pred_loss + loss_lambda * orth_reg_Loss

            loss.backward()
            model.optimizer.step()

            loss_list.append(result_dict['loss'].detach().cpu().data.numpy())
            output_dict['check'] = result_dict['check']

            # update predictor and discriminator
            if len(masked_pred_label) != 0:
                for predictor, label in masked_pred_label:
                    predictor.optimizer.zero_grad()
                    pred_loss = predictor(u_learner_vectors.detach(), label)
                    pred_loss.backward(retain_graph=False)
                    predictor.optimizer.step()

            if len(masked_disc_label) != 0:
                for discriminator, label in masked_disc_label:
                    discriminator.optimizer.zero_grad()
                    disc_loss = discriminator(u_filter_vectors.detach(), label)
                    disc_loss.backward(retain_graph=False)
                    discriminator.optimizer.step()

            # collect predictor evaluation results
            if pred_eval_dict is None:
                pred_eval_dict = self._eval_discriminator(model, labels, u_learner_vectors.detach(), fair_pred_dict, len(mask))
            else:
                pred_batch_eval_dict = self._eval_discriminator(model, labels, u_learner_vectors.detach(), fair_pred_dict, len(mask))
                for f_name in pred_eval_dict:
                    new_label = pred_batch_eval_dict[f_name]['label']
                    current_label = pred_eval_dict[f_name]['label']
                    pred_eval_dict[f_name]['label'] = torch.cat((current_label, new_label), dim=0)

                    new_prediction = pred_batch_eval_dict[f_name]['prediction']
                    current_prediction = pred_eval_dict[f_name]['prediction']
                    pred_eval_dict[f_name]['prediction'] = torch.cat((current_prediction, new_prediction), dim=0)


            # collect discriminator evaluation results
            if disc_eval_dict is None:
                disc_eval_dict = self._eval_discriminator(model, labels, u_filter_vectors.detach(), fair_disc_dict, len(mask))
            else:
                disc_batch_eval_dict = self._eval_discriminator(model, labels, u_filter_vectors.detach(), fair_disc_dict, len(mask))
                for f_name in disc_eval_dict:
                    new_label = disc_batch_eval_dict[f_name]['label']
                    current_label = disc_eval_dict[f_name]['label']
                    disc_eval_dict[f_name]['label'] = torch.cat((current_label, new_label), dim=0)

                    new_prediction = disc_batch_eval_dict[f_name]['prediction']
                    current_prediction = disc_eval_dict[f_name]['prediction']
                    disc_eval_dict[f_name]['prediction'] = torch.cat((current_prediction, new_prediction), dim=0)
            

        # generate predictor evaluation scores
        p_score_dict = {}
        if pred_eval_dict is not None:
            for f_name in pred_eval_dict:
                l = pred_eval_dict[f_name]['label']
                pred = pred_eval_dict[f_name]['prediction']
                n_class = pred_eval_dict[f_name]['num_class']
                p_score_dict[f_name] = self._disc_eval_method(l, pred, n_class)

        # generate discriminator evaluation scores
        d_score_dict = {}
        if disc_eval_dict is not None:
            for f_name in disc_eval_dict:
                l = disc_eval_dict[f_name]['label']
                pred = disc_eval_dict[f_name]['prediction']
                n_class = disc_eval_dict[f_name]['num_class']
                d_score_dict[f_name] = self._disc_eval_method(l, pred, n_class)

        output_dict['d_score'] = d_score_dict
        output_dict['p_score'] = p_score_dict
        output_dict['loss'] = np.mean(loss_list)
        return output_dict

    def train(self, model, dp_dict, fair_pred_dict, fair_disc_dict, loss_lambda = 0.5, skip_eval=0):
        """
        Train model
        :param model: model obj
        :param dp_dict: Data processors for train valid and test
        :param skip_eval: number of epochs to skip for evaluations
        :param fair_disc_dict: fairness discriminator dictionary
        :return:
        """
        train_data = DataLoader(dp_dict['train'], batch_size=self.batch_size, num_workers=self.num_worker,
                                shuffle=True, collate_fn=dp_dict['train'].collate_fn)
        validation_data = DataLoader(dp_dict['valid'], batch_size=None, num_workers=self.num_worker,
                                     pin_memory=True, collate_fn=dp_dict['test'].collate_fn)

        self._check_time(start=True)  # start time
        try:
            for epoch in range(self.epoch):
                self._check_time()
                output_dict = \
                    self.fit(model, train_data, fair_pred_dict, fair_disc_dict, loss_lambda = loss_lambda, epoch=epoch)
                if self.check_epoch > 0 and (epoch == 1 or epoch % self.check_epoch == 0):
                    self.check(model, output_dict)
                training_time = self._check_time()

                if epoch >= skip_eval:
                    valid_result = self.evaluate(model, validation_data) if \
                        validation_data is not None else [-1.0] * len(self.metrics)
                    testing_time = self._check_time()

                    # self.train_results.append(train_result)
                    self.valid_results.append(valid_result)
                    self.pred_results.append(output_dict['p_score'])
                    self.disc_results.append(output_dict['d_score'])

                    logging.info("Epoch %5d [%.1f s]\n validation= %s [%.1f s] "
                                 % (epoch + 1, training_time,
                                    format_metric(valid_result),
                                    testing_time) + ','.join(self.metrics))

                    if best_result(self.metrics[0], self.valid_results) == self.valid_results[-1]:
                        model.save_model()
                        for idx in fair_pred_dict:
                            fair_pred_dict[idx].save_model()
                        for idx in fair_disc_dict:
                            fair_disc_dict[idx].save_model()

                    if self.eva_termination() and self.early_stop == 1:
                        logging.info(
                            "Early stop at %d based on validation result." % (epoch + 1))
                        break
                if epoch < skip_eval:
                    logging.info("Epoch %5d [%.1f s]" %
                                 (epoch + 1, training_time))

        except KeyboardInterrupt:
            logging.info("Early stop manually")
            save_here = input("Save here? (1/0) (default 0):")
            if str(save_here).lower().startswith('1'):
                model.save_model()
                for idx in fair_disc_dict:
                    fair_disc_dict[idx].save_model()

        # Find the best validation result across iterations
        best_valid_score = best_result(self.metrics[0], self.valid_results)
        best_epoch = self.valid_results.index(best_valid_score)

        # prepare disc result string
        disc_info = self.disc_results[best_epoch]
        disc_info_str = ['{}={:.4f}'.format(
            key, disc_info[key][0]) for key in disc_info]
        disc_info_str = ','.join(disc_info_str)

        # prepare disc result string
        pred_info = self.pred_results[best_epoch]
        pred_info_str = ['{}={:.4f}'.format(
            key, pred_info[key][0]) for key in pred_info]
        pred_info_str = ','.join(pred_info_str)

        logging.info("Best Iter (validation) = %5d\t valid= %s [%.1f s] "
                     % (best_epoch + 1,
                        format_metric(self.valid_results[best_epoch]),
                        self.time[1] - self.time[0]) + ','.join(self.metrics))
        logging.info('disc test: ' + disc_info_str + self.disc_metrics[0] + ', pred test: ' + pred_info_str + self.disc_metrics[0])

        model.load_model()
        for idx in fair_pred_dict:
            fair_pred_dict[idx].load_model()
        for idx in fair_disc_dict:
            fair_disc_dict[idx].load_model()

    def check(self, model, out_dict):
        """
        Check intermediate results
        :param model: model obj
        :param out_dict: output dictionary
        :return:
        """
        check = out_dict
        logging.info(os.linesep)
        for i, t in enumerate(check['check']):
            d = np.array(t[1].detach().cpu())
            logging.info(os.linesep.join(
                [t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]) + os.linesep)

        loss, l2 = check['loss'], model.l2()
        l2 = l2 * self.l2_weight
        l2 = l2.detach()
        logging.info('loss = %.4f, l2 = %.4f' % (loss, l2))

        # for discriminator
        pred_score_dict = out_dict['d_score']
        for feature in pred_score_dict:
            logging.info("%s disc test= %s" % (feature, format_metric(
                pred_score_dict[feature])) + ', '.join(self.disc_metrics))

        # for discriminator
        pred_score_dict = out_dict['p_score']
        for feature in pred_score_dict:
            logging.info("%s pred test= %s" % (feature, format_metric(
                pred_score_dict[feature])) + ', '.join(self.disc_metrics))

    def fit_disc(self, model, batches, fair_disc_dict, epoch=-1, lr_attack=None, l2_attack=None):
        """
        Train the discriminator
        :param model: model instance
        :param batches: train data in batches
        :param fair_disc_dict: fairness discriminator dictionary
        :param epoch: epoch number
        :param lr_attack: attacker learning rate
        :param l2_attack: l2 regularization weight for attacker
        :return: return the output of the last round
        """
        gc.collect()
        torch.cuda.empty_cache()

        for idx in fair_disc_dict:
            discriminator = fair_disc_dict[idx]
            if discriminator.optimizer is None:
                discriminator.optimizer = self._build_optimizer(
                    discriminator, lr=lr_attack, l2_weight=l2_attack)
            discriminator.train()

        output_dict = dict()
        loss_acc = defaultdict(list)

        for batch in tqdm(batches, leave=False, desc='Epoch %5d' % (epoch + 1),
                          ncols=100, mininterval=1):
            mask = [1] * model.num_features
            mask = np.asarray(mask)

            batch = batch_to_gpu(batch)

            labels = batch['features']
            masked_disc_label = \
                self._get_masked_disc(fair_disc_dict, labels, mask)

            # calculate recommendation loss + fair discriminator penalty
            uids = batch['X'] - 1
            vectors = model.filter(model.uid_embeddings(uids))
            output_dict['check'] = []

            # update discriminator
            if len(masked_disc_label) != 0:
                for idx, (discriminator, label) in enumerate(masked_disc_label):
                    discriminator.optimizer.zero_grad()
                    disc_loss = discriminator(vectors.detach(), label)
                    disc_loss.backward()
                    discriminator.optimizer.step()
                    loss_acc[discriminator.name].append(
                        disc_loss.detach().cpu())

        for key in loss_acc:
            loss_acc[key] = np.mean(loss_acc[key])

        output_dict['loss'] = loss_acc
        return output_dict
