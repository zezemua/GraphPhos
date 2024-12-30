import sys
import os
import warnings

import torch
import random
import math

from sklearn.utils import shuffle
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.exceptions import UndefinedMetricWarning

import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def getBalanceDataset(labels, nodes_list):
    pos_list_index = []
    pos_num = 0
    for i, every_label in enumerate(labels):
        if every_label == 1:
            pos_list_index.append(i+1)
            pos_num += 1
        else:
            continue
    train_nodes_set = set(nodes_list)
    pos_index_set = set(pos_list_index)
    random_get_index = list(train_nodes_set-pos_index_set)

    batch_size = 2 * pos_num
    batches = math.ceil(len(nodes_list) / batch_size)
    batches *= 2

    if len(nodes_list) >= batch_size:
        neg_list_index_random = random.sample(random_get_index, pos_num)
        nodes_batch = neg_list_index_random + pos_list_index
    else:
        nodes_batch = nodes_list
    return batches, nodes_batch


def getValBalanceDataset(labels, nodes_list):
    pos_list_index = []
    pos_num = 0
    for i, every_label in enumerate(labels):
        if every_label == 1:
            pos_list_index.append(i+1)
            pos_num += 1
        else:
            continue
    train_nodes_set = set(nodes_list)
    pos_index_set = set(pos_list_index)
    random_get_index = list(train_nodes_set-pos_index_set)

    batch_size = 2 * pos_num

    if len(nodes_list) >= batch_size:
        neg_list_index_random = random.sample(random_get_index, pos_num)
        nodes_batch = neg_list_index_random + pos_list_index
    else:
        nodes_batch = nodes_list
    return nodes_batch


def apply_model(train_nodes_index, seq_len, select_nodes_index, label_list, graphSage, classification, cnn, b_sz, device, learn_method, id, features, aa_adj_dict, features_protbert):
    train_nodes = train_nodes_index
    labels = label_list

    models = [graphSage, classification, cnn]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)

    optimizer = torch.optim.Adam(params, lr=0.001)
    optimizer.zero_grad()

    for model in models:
        model.zero_grad()

    visited_nodes = set()
    batches, nodes_batch = getBalanceDataset(labels, train_nodes)
    for index in range(batches + 1):
        random.shuffle(nodes_batch)

        visited_nodes |= set(nodes_batch)

        labels_batch = []
        for item in nodes_batch:
            labels_batch.append(labels[item-1])
        labels_batch = torch.tensor(labels_batch, dtype=torch.float32).view(-1, 1)

        embs_batch = graphSage(nodes_batch, features, aa_adj_dict, seq_len)
        protbert_embs_batch = cnn(nodes_batch, features_protbert)
        embs_concat = torch.cat([protbert_embs_batch, embs_batch], dim=1)
        logists = classification(embs_concat)

        # weight = torch.tensor([2]).to(device)
        loss_function = torch.nn.BCELoss().to(device)
        loss_sup = loss_function(logists, labels_batch.to(device))
        loss = loss_sup

        # temp_loss_result = "/home/data/b532wangzeyu/MyGraphSageTest/epoch20_train_loss/" + str(id)
        # with open(temp_loss_result, 'a+') as fileloss:
        #     fileloss.writelines(id + " Step[" + str(index+1) + "/" + str(batches) + "], Loss: " + str(round(loss.item(), 4)) + "\n")
        # print('[{}] Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(id, index + 1, batches, loss.item(), len(visited_nodes), len(train_nodes)))

        loss.backward()

        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()
        optimizer.zero_grad()

        for model in models:
            model.zero_grad()

    return graphSage, classification, cnn


def evaluate(id, test_nodes_set, val_nodes_set, label_list, graphSage, classification, cnn, device, max_vali_f1, name, cur_epoch, features, aa_adj_dict, features_protbert, seq_len):
    test_nodes = test_nodes_set
    labels = label_list
    loss = 0
    # print(id)
    if len(test_nodes) != 0 and len(val_nodes_set) != 0:
        models = [graphSage, classification, cnn]
        params = []

        for model in models:
            for param in model.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    params.append(param)
        val_nodes = getValBalanceDataset(labels, val_nodes_set)
        embs = graphSage(val_nodes, features, aa_adj_dict, seq_len)
        protbert_embs_batch = cnn(val_nodes, features_protbert)
        embs_concat = torch.cat([protbert_embs_batch, embs], dim=1)
        logists = classification(embs_concat)

        labels_val = []
        for val_node in val_nodes:
            labels_val.append(labels[val_node - 1])

        labels_val = torch.tensor(labels_val, dtype=torch.float32).view(-1, 1)

        # weight = torch.tensor([2]).to(device)
        loss_function = torch.nn.BCELoss().to(device)
        loss_sup = loss_function(logists, labels_val.to(device))

        loss = loss + loss_sup

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)

            threshold = 0.5
            predicts = (logists > threshold).int()

            assert len(labels_val) == len(predicts)

            vali_f1 = f1_score(labels_val, predicts.cpu(), average="weighted")
            if vali_f1 > max_vali_f1 and vali_f1 != 1.0000:
                max_vali_f1 = vali_f1
                embs = graphSage(test_nodes, features, aa_adj_dict, seq_len)
                protbert_embs_batch = cnn(test_nodes, features_protbert)
                embs_concat = torch.cat([protbert_embs_batch, embs], dim=1)
                logists = classification(embs_concat)

                threshold = 0.5
                predicts = (logists > threshold).int()

                labels_test = []
                for test_node in test_nodes:
                    labels_test.append(labels[test_node - 1])

                predicts = predicts.cpu()
                assert len(labels_test) == len(predicts)

                test_f1 = f1_score(labels_test, predicts, average="weighted")

                for param in params:
                    param.requires_grad = True
                #
                # if test_f1 > 0.5:
                torch.save(models, 'STY_models_af_0305/model_best_{}_ep{}_{:.4f}.torch'.format(name, cur_epoch, test_f1))
            elif vali_f1 == 1.0000:
                vali_f1 = max_vali_f1

        for param in params:
            param.requires_grad = True
    loss_aver = loss / len(val_nodes_set)
    return max_vali_f1, loss_aver


