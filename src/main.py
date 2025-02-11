# coding=utf-8
import sys
import os
from utils.generic import *
from data_reader import RecDataReader
from runner import *
from datasets import RecDataset
from models.PMF import *
from models.MLP import *
from models.BiasedMF import *
from models.DMF import *
from models.BaseRecModel import BaseRecModel
from models.Discriminators import BinaryDiscriminator
from data_reader import DiscriminatorDataReader
from datasets import DiscriminatorDataset
from torch.utils.data import DataLoader
from rankingdata_loader import RankingDataLoader



def main():
    # init args
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--data_reader', type=str, default='RecDataReader',
                             help='Choose data_reader')
    init_parser.add_argument('--data_processor', type=str, default='RecDataset',
                             help='Choose data_processor')
    init_parser.add_argument('--model_name', type=str, default='BiasedMF',
                             help='Choose model to run.')
    init_parser.add_argument('--fairness_framework', type=str, default='None',
                             help='Choose fairness framework, e.g., PCFR, FairRec, FOCF_BprLoss, FOCF_AbsUnf, FOCF_ValUnf, SolutionA.')
    init_parser.add_argument('--runner', type=str, default='RecRunner',
                             help='Choose runner')
    init_args, init_extras = init_parser.parse_known_args()

    # choose data_reader
    data_reader_name = eval(init_args.data_reader)

    fairness_framework = init_args.fairness_framework
    # choose model
    if init_args.fairness_framework != 'None':
        model_name = eval(init_args.model_name + '_' + init_args.fairness_framework)
    else:
        model_name = eval(init_args.model_name)

    if init_args.fairness_framework != 'None':
        runner_name = eval(init_args.runner + '_' + init_args.fairness_framework)
    else:
        runner_name = eval(init_args.runner)

    # choose data_processor
    data_processor_name = eval(init_args.data_processor)

    # cmd line paras
    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = data_reader_name.parse_data_args(parser)
    parser = BinaryDiscriminator.parse_disc_args(parser)
    parser = model_name.parse_model_args(parser, model_name=init_args.model_name)
    parser = runner_name.parse_runner_args(parser)
    parser = data_processor_name.parse_dp_args(parser)
    parser = DiscriminatorDataset.parse_dp_args(parser)

    args, extras = parser.parse_known_args()

    # log,model, result filename
    log_file_name = [init_args.model_name + '_' + init_args.fairness_framework,
                     args.dataset + '_' + '_'.join(args.feature_columns), str(args.random_seed),
                     'optimizer=' + args.optimizer, 'lr=' + str(args.lr), 'l2=' + str(args.l2),
                     'dropout=' + str(args.dropout), 'batch_size=' + str(args.batch_size)]
    log_file_name = '__'.join(log_file_name).replace(' ', '__')
    if args.log_file == '../log/log.txt':
        args.log_file = '../log/%s.txt' % log_file_name
    if args.model_path == '../model/%s/%s.pt' % (init_args.model_name, init_args.model_name):
        args.model_path = '../model/%s/%s.pt' % (init_args.model_name, log_file_name)

    # logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # convert the namespace into dictionary e.g. init_args.model_name -> {'model_name': BaseModel}
    logging.info(vars(init_args))
    logging.info(vars(args))

    logging.info('DataReader: ' + init_args.data_reader)
    logging.info('Model: ' + init_args.model_name)
    logging.info('Fairness framework: ' + init_args.fairness_framework)
    logging.info('Runner: ' + init_args.runner)
    logging.info('DataProcessor: ' + init_args.data_processor)

    # random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)

    # cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logging.info("# cuda devices: %d" % torch.cuda.device_count())
    # create data_reader
    data_reader = data_reader_name(path=args.path, dataset_name=args.dataset, feature_columns=args.feature_columns, sep=args.sep)

    # create data processor
    data_processor_dict = {}
    for stage in ['train', 'valid', 'test']:
        if stage == 'train':
            if init_args.data_processor in ['RecDataset']:
                data_processor_dict[stage] = data_processor_name(
                    data_reader, stage, batch_size=args.batch_size, num_neg=args.train_num_neg)
            else:
                raise ValueError('Unknown DataProcessor')
        else:
            if init_args.data_processor in ['RecDataset']:
                data_processor_dict[stage] = data_processor_name(
                    data_reader, stage, batch_size=args.vt_batch_size, num_neg=args.vt_num_neg)
            else:
                raise ValueError('Unknown DataProcessor')

    # create model with or without fairness framework
    if init_args.model_name in ['BiasedMF', 'PMF']:
            model = model_name(data_processor_dict, user_num=len(data_reader.user_ids_set),
                           item_num=len(data_reader.item_ids_set), u_vector_size=args.u_vector_size,
                           i_vector_size=args.i_vector_size, random_seed=args.random_seed, dropout=args.dropout,
                           model_path=args.model_path)
    elif init_args.model_name in ['DMF', 'MLP']:
        model = model_name(data_processor_dict, user_num=len(data_reader.user_ids_set),
                           item_num=len(data_reader.item_ids_set), u_vector_size=args.u_vector_size,
                           i_vector_size=args.i_vector_size, num_layers=args.num_layers,
                           random_seed=args.random_seed, dropout=args.dropout,
                           model_path=args.model_path)
    else:
        logging.error('Unknown Model: ' + init_args.model_name)
        return
    # init model paras
    model.apply(model.init_weights)

    # use gpu
    if torch.cuda.device_count() > 0:
        model = model.cuda()

    
    if init_args.fairness_framework in ['PCFR', 'SolutionA']:
        # create discriminators
        fair_disc_dict = {}
        for feat_idx in data_reader.feature_info:
            fair_disc_dict[feat_idx] = \
                BinaryDiscriminator(args.u_vector_size, data_reader.feature_info[feat_idx],
                            random_seed=args.random_seed, dropout=args.dropout, neg_slope=args.neg_slope,
                            model_dir_path=os.path.dirname(args.model_path), layers=args.discriminator_layers)
            fair_disc_dict[feat_idx].apply(fair_disc_dict[feat_idx].init_weights)
            if torch.cuda.device_count() > 0:
                fair_disc_dict[feat_idx] = fair_disc_dict[feat_idx].cuda()
    elif init_args.fairness_framework in ['FairRec']:
        # create discriminators
        fair_disc_dict = {}
        for feat_idx in data_reader.feature_info:
            fair_disc_dict[feat_idx] = \
                BinaryDiscriminator(args.u_vector_size, data_reader.feature_info[feat_idx],
                            random_seed=args.random_seed, dropout=args.dropout, neg_slope=args.neg_slope,
                            model_dir_path=os.path.dirname(args.model_path), layers=args.discriminator_layers)
            fair_disc_dict[feat_idx].apply(fair_disc_dict[feat_idx].init_weights)
            if torch.cuda.device_count() > 0:
                fair_disc_dict[feat_idx] = fair_disc_dict[feat_idx].cuda()

        # create predictors
        fair_pred_dict = {}
        for feat_idx in data_reader.feature_info:
            fair_pred_dict[feat_idx] = \
                BinaryDiscriminator(args.u_vector_size, data_reader.feature_info[feat_idx],
                            random_seed=args.random_seed, dropout=args.dropout, neg_slope=args.neg_slope,
                            model_dir_path=os.path.dirname(args.model_path), layers=args.predictor_layers)
            fair_pred_dict[feat_idx].apply(fair_pred_dict[feat_idx].init_weights)
            if torch.cuda.device_count() > 0:
                fair_pred_dict[feat_idx] = fair_pred_dict[feat_idx].cuda()

    assert len(args.feature_columns) == 1
    all_df = pd.read_csv(os.path.join('../dataset/'+ args.dataset, args.dataset + '.all.tsv'),
                    header=0, sep='\t',
                    usecols=['uid', args.feature_columns[0]], engine='python')
    group_0_ids = set(all_df[all_df[args.feature_columns[0]] == 0]['uid'].unique())
    group_1_ids = set(all_df[all_df[args.feature_columns[0]] == 1]['uid'].unique())
    
    # create runner
    # batch_size is the training batch size, eval_batch_size is the batch size for evaluation
    if init_args.runner in ['RecRunner']:
        if init_args.fairness_framework in ['FOCF_ValUnf', 'FOCF_AbsUnf', 'SolutionA']:
            runner = runner_name(
                group_0_ids = group_0_ids, group_1_ids = group_1_ids,
                optimizer=args.optimizer, learning_rate=args.lr,
                epoch=args.epoch, batch_size=args.batch_size, eval_batch_size=args.vt_batch_size,
                dropout=args.dropout, l2=args.l2,
                metrics=args.metric, disc_metrics = args.disc_metric, check_epoch=args.check_epoch, early_stop=args.early_stop, num_worker=args.num_worker, 
                disc_epoch=args.disc_epoch)
        else:
            runner = runner_name(
                optimizer=args.optimizer, learning_rate=args.lr,
                epoch=args.epoch, batch_size=args.batch_size, eval_batch_size=args.vt_batch_size,
                dropout=args.dropout, l2=args.l2,
                metrics=args.metric, disc_metrics = args.disc_metric, check_epoch=args.check_epoch, early_stop=args.early_stop, num_worker=args.num_worker, 
                disc_epoch=args.disc_epoch)
    else:
        logging.error('Unknown Runner: ' + init_args.runner)
        return

    if args.load > 0:
        model.load_model()
        if init_args.fairness_framework == 'PCFR' or init_args.fairness_framework == 'SolutionA':
            for idx in fair_disc_dict:
                fair_disc_dict[idx].load_model()
        if init_args.fairness_framework == 'FairRec':
            for idx in fair_disc_dict:
                fair_disc_dict[idx].load_model()
            for idx in fair_pred_dict:
                fair_pred_dict[idx].load_model()

    if args.train > 0:
        if fairness_framework in ['None', 'FOCF_AbsUnf', 'FOCF_BprLoss', 'FOCF_ValUnf']:
            runner.train(model, data_processor_dict, skip_eval = args.skip_eval)
        elif fairness_framework in ['PCFR', 'SolutionA']:
            runner.train(model, data_processor_dict, fair_disc_dict = fair_disc_dict, skip_eval = args.skip_eval)
        elif fairness_framework in ['FairRec']:
            runner.train(model, data_processor_dict, fair_pred_dict = fair_pred_dict, fair_disc_dict = fair_disc_dict, loss_lambda = args.fairrec_lambda, skip_eval = args.skip_eval)

    # reset seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)

    if args.eval_disc:
        # Train extra discriminator for evaluation
        # create data reader
        disc_data_reader = DiscriminatorDataReader(path=args.path, dataset_name=args.dataset, feature_columns=args.feature_columns, sep=args.sep, test_ratio=0.2)

        # create data processor
        extra_data_processor_dict = {}
        for stage in ['train', 'test']:
            extra_data_processor_dict[stage] = DiscriminatorDataset(disc_data_reader, stage, args.disc_batch_size)

        # create discriminators
        extra_fair_disc_dict = {}
        for feat_idx in disc_data_reader.feature_info:
            if disc_data_reader.feature_info[feat_idx].num_class == 2:
                extra_fair_disc_dict[feat_idx] = \
                    BinaryDiscriminator(args.u_vector_size, disc_data_reader.feature_info[feat_idx],
                                   random_seed=args.random_seed, dropout=args.dropout,
                                   neg_slope=args.neg_slope, model_dir_path=os.path.dirname(args.model_path),
                                   model_name='eval', layers=args.attacker_layers)
            else:
                raise ValueError('Not support multi-class features')
            if torch.cuda.device_count() > 0:
                extra_fair_disc_dict[feat_idx] = extra_fair_disc_dict[feat_idx].cuda()

        if args.load_attack:
            for idx in extra_fair_disc_dict:
                logging.info('load attacker model...')
                extra_fair_disc_dict[idx].load_model()
        model.load_model()
        model.freeze_model()
        runner.train_discriminator(model, extra_data_processor_dict, extra_fair_disc_dict, args.lr_attack,
                                   args.l2_attack)

    test_data = DataLoader(data_processor_dict['test'], batch_size=None, num_workers=args.num_worker,
                           pin_memory=True, collate_fn=data_processor_dict['test'].collate_fn)


    test_result = runner.evaluate(model, test_data)
    logging.info("Test After Training = %s "
                    % (format_metric(test_result)) + ','.join(runner.metrics))

    ranking_result = runner.generate_rank_file(model, test_data)
    rank_dl = RankingDataLoader(ranking_result = ranking_result, g1_user_list = group_0_ids, g2_user_list = group_1_ids)
    all_df = rank_dl.rank_df.copy(deep=True)
    group_df_list = [rank_dl.g1_df.copy(deep=True),
                        rank_dl.g2_df.copy(deep=True)]
    logging.info('Value unfairness: ' + str(value_unfairness(group_df_list[0], group_df_list[1])))
    logging.info('Absolute unfairness: ' + str(absolute_unfairness(group_df_list[0], group_df_list[1])))
    logging.info('User-oriented unfairness: ' + str(user_oriented_unfairness(group_df_list[0], group_df_list[1], runner.metrics[0])))
    logging.info('Calibrated group-wise utility: ' + str(calibrated_groupwise_utility(group_df_list[0], group_df_list[1], runner.metrics[0])))
    
    return


if __name__ == '__main__':
    main()
