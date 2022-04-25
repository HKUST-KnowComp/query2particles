import argparse
import gc
import json
import logging
import os
import random

import numpy as np
import numpy
import torch

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator


from model import Query2Particles
from dataloader import *
from tensorboardX import SummaryWriter
import time
import pickle
import collections
import datetime

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
try:
    from apex import amp
except:
    print("apex not installed")

query_name_dict = {('e', ('r',)): '1p',
                       ('e', ('r', 'r')): '2p',
                       ('e', ('r', 'r', 'r')): '3p',
                       (('e', ('r',)), ('e', ('r',))): '2i',
                       (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                       ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                       (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                       (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                       (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                       ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                       (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                       (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                       (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                       ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                       ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                       ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                       }
name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(
    name_query_dict.keys())  # ['1p', '2p',
# '3p', '2i', '3i',
# 'ip', 'pi', '2in', '3in', 'inp', 'pin', 'pni', '2u-DNF',
# '2u-DM', 'up-DNF', 'up-DM']



def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True



def do_train_step(model, train_iterators, optimizer, use_apex, step):

    # training
    logs = []

    for i in range(20):

        probability_dict = {"1p": 3, "2p": 1.5, "3p": 1.5,
                            "2i": 1, "3i": 1,
                            "2in": 0.1, "3in": 0.1,
                            "inp": 0.1, "pni": 0.1,
                            "pin": 0.1}

        iteration_list = list(train_iterators.items())

        probability_list = [probability_dict[task_name] for task_name, iter in iteration_list]
        task_name_list = [task_name for task_name, iter in iteration_list]

        probability_list = numpy.array(probability_list)
        probability_list = probability_list / numpy.sum(probability_list)

        task_name = np.random.choice(task_name_list, p=probability_list)

        iter = train_iterators[task_name]

        if torch.cuda.device_count() > 1:
            log_of_the_step = model.module.train_step(model, iter, optimizer, use_apex)
        else:
            log_of_the_step = model.train_step(model, iter, optimizer, use_apex)

        # Need to write summary here

        logs.append(log_of_the_step)

        if task_name in ["1p"]:

            if torch.cuda.device_count() > 1:
                log_of_the_step = model.module.train_inv_step(model, iter, optimizer, use_apex)
            else:
                log_of_the_step = model.train_inv_step(model, iter, optimizer, use_apex)

            logs.append(log_of_the_step)

    return logs



def do_test(model, test_dataset_iter, summary_writer, step, easy_answers, hard_answers):

    # This function will do the evaluation as a whole,
    # not do it in a step-wise way.
    all_average_dev_logs = {}
    all_average_test_logs = {}

    use_cuda = True

    for task_name, iter in test_dataset_iter.items():
        # qtype, iter = iter

        aggregated_logs = {}

        early_stop = False
        counter = 0

        for negative_sample, queries, queries_unflatten, query_structures  in tqdm(iter):

            if early_stop:
                counter += 1
                if counter > 3:
                    break

            if torch.cuda.device_count() > 1:
                log_of_the_step, _ = model.module.evaluate_step(model,
                                                             negative_sample,
                                                             queries,
                                                             queries_unflatten,
                                                             query_structures,
                                                             easy_answers,
                                                             hard_answers)
            else:
                log_of_the_step, _ = model.evaluate_step(model,
                                                      negative_sample,
                                                      queries,
                                                      queries_unflatten,
                                                      query_structures,
                                                      easy_answers,
                                                      hard_answers)

            for key, value in log_of_the_step.items():

                if key in aggregated_logs:
                    aggregated_logs[key].append(value)
                else:
                    aggregated_logs[key] = [value]

        aggregated_weighted_mean_logs = {}

        for key, value in aggregated_logs.items():
            if key != "num_samples":
                weighted_average = np.sum( np.array(value) *
                                           np.array(aggregated_logs["num_samples"]) ) / \
                                   np.sum(aggregated_logs["num_samples"])

                aggregated_weighted_mean_logs[key] = weighted_average
                summary_writer.add_scalar(task_name+ "_" + key, weighted_average, step)


        if "test" in task_name:

            if "num_samples" in all_average_test_logs:
                all_average_test_logs["num_samples"].append(
                    np.sum(aggregated_logs["num_samples"])
                )

            else:
                all_average_test_logs["num_samples"] = [np.sum(aggregated_logs["num_samples"])]


            for key, value in aggregated_weighted_mean_logs.items():
                if key in all_average_test_logs:
                    all_average_test_logs[key].append(value)
                else:
                    all_average_test_logs[key] = [value]




        elif "dev" in task_name:
            if "num_samples" in all_average_dev_logs:
                all_average_dev_logs["num_samples"].append(
                    np.sum(aggregated_logs["num_samples"])
                )

            else:
                all_average_dev_logs["num_samples"] = [np.sum(aggregated_logs["num_samples"])]

            for key, value in aggregated_weighted_mean_logs.items():
                if key in all_average_dev_logs:
                    all_average_dev_logs[key].append(value)
                else:
                    all_average_dev_logs[key] = [value]

    weighted_average_dev_logs = {}
    weighted_average_test_logs = {}


    for key, value in all_average_test_logs.items():
        if key != "num_samples":
            weighted_average = np.sum(np.array(value) *
                                      np.array(all_average_test_logs["num_samples"])) / \
                               np.sum(all_average_test_logs["num_samples"])

            weighted_average_test_logs[key] = weighted_average
            summary_writer.add_scalar("_average_test_" + key, weighted_average, step)

    for key, value in all_average_dev_logs.items():
        if key != "num_samples":
            weighted_average = np.sum(np.array(value) *
                                      np.array(all_average_dev_logs["num_samples"])) / \
                               np.sum(all_average_dev_logs["num_samples"])

            weighted_average_dev_logs[key] = weighted_average
            summary_writer.add_scalar("_average_dev_" + key, weighted_average, step)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-n', '--negative_sample_size', default=1, type=int)

    parser.add_argument('--log_steps', default=6000, type=int, help='train log every xx steps')
    parser.add_argument('--data_name', type=str, required=True)

    parser.add_argument('-d', '--entity_space_dim', default=400, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.002, type=float)
    parser.add_argument('-wc', '--weight_decay', default=0.0000, type=float)

    parser.add_argument('-p', '--num_particles', default=2, type=int)

    parser.add_argument("--model_name", default="v46.xx", type=str)

    # parser.add_argument('--entity_embeddings', type=str, required=True)
    # parser.add_argument('--relation_embeddings', type=str, required=True)
    # parser.add_argument("--trainable_embeddings", action='store_true')
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--dropout_rate", default=0.1, type=float)
    parser.add_argument('-ls', "--label_smoothing", default=0.0, type=float)
    parser.add_argument("--warm_up_steps", default=100, type=int)

    parser.add_argument("--checkpoint_path", type=str, default="")


    args = parser.parse_args()
    print(args)

    use_apex = False
    # load data first
    max_train_steps = 300000000
    previous_steps = 0
    # data_name = "FB15k-237-betae"
    # data_name = "FB15k-betae"
    # data_name = "NELL-betae"

    data_name = args.data_name
    data_path = "../data/" + data_name
    with open('%s/stats.txt' % data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = './q2p_logs/gradient_tape/' + current_time + data_name + '/train'
    test_log_dir = './q2p_logs/gradient_tape/' + current_time + data_name + '/test'
    train_summary_writer = SummaryWriter(train_log_dir)
    test_summary_writer = SummaryWriter(test_log_dir)

    # print("Load pre_trained embeddings")

    # with open("../data/" + args.entity_embeddings, 'rb') as f:
    #     ent_embedding = np.load(f)
    #
    # with open("../data/" + args.relation_embeddings, 'rb') as f:
    #     rel_embedding = np.load(f)

    # ent_embedding = np.load("../data/" + args.entity_embeddings, 'r')
    # rel_embedding = np.load("../data/" + args.relation_embeddings, 'r')

    model = Query2Particles(nentity,
                            nrelation,
                            args.entity_space_dim,
                            num_particles=args.num_particles,
                            dropout_rate=args.dropout_rate,
                            label_smoothing=args.label_smoothing)

    # Load the model with the given path
    if args.checkpoint_path != "":
        checkpoint_path = args.checkpoint_path
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        previous_steps = checkpoint['steps']

    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)


    # PATH = "logs/20201117-123347"
    # model.load_state_dict(torch.load(PATH))

    model.cuda()

    no_decay = ['bias', 'layer_norm', 'embedding', "off_sets"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.optimizer == "adam":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

    else:
        print("invalid optimizer name, using adam instead")
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

    # Load the model with the given path
    if args.checkpoint_path != "":
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    use_apex = torch.cuda.device_count() == 1 and use_apex

    if use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    def warmup_lambda(epoch):
        if epoch < args.warm_up_steps:
            return epoch * 1.0 / args.warm_up_steps
        else:
            return 1

    scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # task = "1c.2c.3c.2i.3i.ic.ci.2u.uc"
    #
    # tasks = task.split('.')
    #
    # all_train_ans = dict()
    # all_valid_ans = dict()
    # all_valid_ans_hard = dict()
    # all_test_ans = dict()
    # all_test_ans_hard = dict()
    #
    # train_tasks = ['1c', '2c', '3c', '2i', '3i']
    # evaluate_only_tasks = ['ic', 'ci', '2u', 'uc']
    # supported_tasks = train_tasks + evaluate_only_tasks
    #
    # all_train_triples = {}
    # all_valid_triples = {}
    # all_test_triples = {}
    #
    batch_size = args.batch_size
    negative_sample_size = args.negative_sample_size
    cpu_num = 4

    test_batch_size = 1

    train_iterators = {}
    test_iterators = {}
    dev_iterators = {}

    def load_data():
        '''
        Load all queries
        '''
        print("loading data")
        train_queries = pickle.load(open(os.path.join(data_path, "train-queries.pkl"), 'rb'))
        train_answers = pickle.load(open(os.path.join(data_path, "train-answers.pkl"), 'rb'))
        valid_queries = pickle.load(open(os.path.join(data_path, "valid-queries.pkl"), 'rb'))
        valid_hard_answers = pickle.load(open(os.path.join(data_path, "valid-hard-answers.pkl"), 'rb'))
        valid_easy_answers = pickle.load(open(os.path.join(data_path, "valid-easy-answers.pkl"), 'rb'))
        test_queries = pickle.load(open(os.path.join(data_path, "test-queries.pkl"), 'rb'))
        test_hard_answers = pickle.load(open(os.path.join(data_path, "test-hard-answers.pkl"), 'rb'))
        test_easy_answers = pickle.load(open(os.path.join(data_path, "test-easy-answers.pkl"), 'rb'))


        return train_queries, train_answers, valid_queries,\
               valid_hard_answers, valid_easy_answers, test_queries, \
               test_hard_answers, test_easy_answers


    train_queries, train_answers, valid_queries, valid_hard_answers,\
    valid_easy_answers, test_queries, test_hard_answers, test_easy_answers = load_data()

    for query_structure in train_queries:
        print(query_name_dict[query_structure] + ": " + str(len(train_queries[query_structure])))



    print("====== Create Training Iterators  ======")
    for query_structure in train_queries:
        tmp_queries = list(train_queries[query_structure])
        tmp_queries = [(query, query_structure) for query in tmp_queries]
        new_iterator = SingledirectionalOneShotIterator(DataLoader(
            TrainDataset(tmp_queries, nentity, nrelation, negative_sample_size, train_answers),
            batch_size=batch_size,
            shuffle=True,
            num_workers=cpu_num,
            collate_fn=TrainDataset.collate_fn
        ))
        train_iterators[query_name_dict[query_structure]] = new_iterator

        # print(query_name_dict[query_structure])

    print("====== Create Testing Dataloader ======")
    for query_structure in test_queries:
        logging.info(query_name_dict[query_structure] + ": " + str(len(test_queries[query_structure])))

        tmp_queries = list(test_queries[query_structure])
        tmp_queries = [(query, query_structure) for query in tmp_queries]
        test_dataloader = DataLoader(
            TestDataset(
                tmp_queries,
                nentity,
                nrelation,
            ),
            batch_size=test_batch_size,
            num_workers=cpu_num,
            collate_fn=TestDataset.collate_fn
        )

        # print("test_" + query_name_dict[query_structure])
        test_iterators["test_" + query_name_dict[query_structure]] = test_dataloader


    print("====== Create Validation Dataloader ======")

    for query_structure in valid_queries:
        logging.info(query_name_dict[query_structure] + ": " + str(len(valid_queries[query_structure])))

        tmp_queries = list(valid_queries[query_structure])
        tmp_queries = [(query, query_structure) for query in tmp_queries]
        valid_dataloader = DataLoader(
            TestDataset(
                tmp_queries,
                nentity,
                nrelation,
            ),
            batch_size=test_batch_size,
            num_workers=cpu_num,
            collate_fn=TestDataset.collate_fn
        )
        # print("dev_" + query_name_dict[query_structure])
        dev_iterators["dev_" + query_name_dict[query_structure]] = valid_dataloader

    for step in tqdm(range(max_train_steps)):

        total_step = step + previous_steps
        logs = do_train_step(model, train_iterators, optimizer, use_apex, total_step)
        scheduler.step()

        aggregated_logs = {}
        for log in logs:

            for key, value in log.items():
                if key == "qtype":
                    continue
                train_summary_writer.add_scalar(log["qtype"]+ "_" + key, value, total_step)

                if key in aggregated_logs:
                    aggregated_logs[key].append(value)
                else:
                    aggregated_logs[key] = [value]

        aggregated_mean_logs = {key: np.mean(value) for key, value in aggregated_logs.items()}

        for key, value in aggregated_mean_logs.items():

            train_summary_writer.add_scalar( "_average_" + key, value, total_step)

        save_step = args.log_steps
        model_name = args.model_name
        if step % save_step == 0:
            general_checkpoint_path = "./q2p_logs/" + model_name +"_"+ str(total_step) +"_"+ data_name +".bin"

            if torch.cuda.device_count() > 1:
                torch.save({
                    'steps': total_step,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, general_checkpoint_path)
            else:
                torch.save({
                    'steps': total_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, general_checkpoint_path)

        if step % save_step == 0 and step > 0:
            do_test(model, test_iterators, test_summary_writer, total_step, test_easy_answers, test_hard_answers)
            do_test(model, dev_iterators, test_summary_writer, total_step, valid_easy_answers, valid_hard_answers)

        if step % 50 == 0:
            gc.collect()

if __name__ == "__main__":
    main()
