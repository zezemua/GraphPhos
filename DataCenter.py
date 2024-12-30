import argparse
import csv
import random
from collections import defaultdict

from tqdm import tqdm
import pyhocon
import numpy as np


def _split_data(nodesSTY_list, labelSTY_list, nodes_num, test_split=5, val_split=5):
    pos_num = labelSTY_list.count(1)
    pos_index = []
    for i, pos_label in enumerate(labelSTY_list):
        if pos_label == 1:
            pos_index.append(nodesSTY_list[i])
        else:
            continue
    nodes_set = set(nodesSTY_list)
    nodes_pos_set = set(pos_index)
    nodes_neg_list = list(nodes_set-nodes_pos_set)

    nodes_num = nodes_num - pos_num
    rand_indices = np.random.permutation(nodes_num)
    test_size = nodes_num // test_split
    val_size = nodes_num // val_split
    train_size = nodes_num - (test_size + val_size)

    test_indexs = rand_indices[:test_size]
    val_indexs = rand_indices[test_size:(test_size + val_size)]
    train_indexs = rand_indices[(test_size + val_size):]

    train_nodes = []
    test_nodes = []
    val_nodes = []
    for item_train in train_indexs:
        train_nodes.append(nodes_neg_list[item_train])
    for item_test in test_indexs:
        test_nodes.append(nodes_neg_list[item_test])
    for item_val in val_indexs:
        val_nodes.append(nodes_neg_list[item_val])

    rand_indices_pos = np.random.permutation(pos_num)
    test_size_pos = pos_num // test_split
    val_size_pos = pos_num // val_split
    train_size_pos = pos_num - (test_size + val_size)

    test_indexs_pos = rand_indices_pos[:test_size_pos]
    val_indexs_pos = rand_indices_pos[test_size_pos:(test_size_pos + val_size_pos)]
    train_indexs_pos = rand_indices_pos[(test_size_pos + val_size_pos):]

    for item_train_pos in train_indexs_pos:
        train_nodes.append(pos_index[item_train_pos])
    for item_test_pos in test_indexs_pos:
        test_nodes.append(pos_index[item_test_pos])
    for item_val_pos in val_indexs_pos:
        val_nodes.append(pos_index[item_val_pos])

    return test_nodes, val_nodes, train_nodes


def data_load():
    train_id_list = []
    with open("/home/data/b532wangzeyu/MyGraphSageTest/Train_data/STY_train", 'r') as fasta_ids_read:
        ids = fasta_ids_read.readlines()
        for id_line in ids:
            id = id_line.strip()
            train_id_list.append(id)
    return train_id_list


def test_data_load():
    train_id_list = []
    with open("/home/data/b532wangzeyu/MyGraphSageTest/Test_data/STY_test", 'r') as fasta_ids_read:
        ids = fasta_ids_read.readlines()
        for id_line in ids:
            id = id_line.strip()
            train_id_list.append(id)
    return train_id_list


def char_map(id):
    char_dict = {}
    fasta_filename = "/home/data/b532wangzeyu/MyGraphSageTest/fastas/" + id + ".fasta"
    with open(fasta_filename, 'r') as fileread:
        head = fileread.readline()
        seq_len = int(head.split(' ')[1]) - 1
        seq = fileread.readline().strip()

        for index, char in enumerate(seq):
            char_dict[index + 1] = char

    return seq_len, seq, char_dict


def readContent(id):
    feature_list = []
    label_list = []
    phos_content_file = "/home/data/b532wangzeyu/MyGraphSageTest/concat_features/" + id
    with open(phos_content_file) as fpcontent:
        for i, line in enumerate(fpcontent):
            info = line.strip().split(', ')
            feature_list.append([float(x) for x in info[1:-1]])
            label_list.append(int(info[-1]))

    feature_list = np.asarray(feature_list)
    label_list = np.asarray(label_list, dtype=np.int64)
    return feature_list, label_list


def readCites(id):
    # phos_cites_file = "/home/data/b532wangzeyu/MyGraphSageTest/cite_features/" + id + ".cm"
    phos_cites_file = "/home/data/b532wangzeyu/MyGraphSageTest/contact_predict/" + id + ".cm"
    aa_adj_dict = defaultdict(set)
    with open(phos_cites_file) as fpcites:
        for i, line in enumerate(fpcites):
            info = line.strip().split()
            assert len(info) == 2
            aa1 = info[0]
            aa2 = info[1]
            aa_adj_dict[aa1].add(aa2)
            aa_adj_dict[aa2].add(aa1)
    return aa_adj_dict


def select_nodes(seq_len, char_dict, feature_list, label_list):
    select_nodes_index = []
    for char_index in range(seq_len):
        real_index = char_index + 1
        # if char_dict[real_index] == 'Y':
        if char_dict[real_index] == 'S' or char_dict[real_index] == 'T' or char_dict[real_index] == 'Y':
            select_nodes_index.append(int(real_index))
        else:
            continue

    featureSTY_list = []
    labelSTY_list = []

    for item in select_nodes_index:
        featureSTY_list.append(feature_list[item - 1])
        labelSTY_list.append(label_list[item - 1])
    featureSTY_list = np.asarray(featureSTY_list)
    # labelSTY_list = np.asarray(labelSTY_list)

    return select_nodes_index, featureSTY_list, labelSTY_list


def read_protbert_feature(id, seq_len):
    protbert_file = "/home/data/b532wangzeyu/MyGraphSageTest/concat_protbert_feature/" + id
    feature_list = []
    with open(protbert_file) as fpcontent:
        for i, line in enumerate(fpcontent):
            info = line.strip().split(',')
            feature_list.append([float(x) for x in info[1:-1]])
    feature_list = np.asarray(feature_list)
    return feature_list
