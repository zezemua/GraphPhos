import torch
import torch.nn as nn
import argparse
import csv
import random
from tqdm import tqdm
import pyhocon
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef
import torch
from sklearn.metrics import roc_curve, auc
from src_STY.DataCenter import *
from src_STY.Models import *
from src_STY.utils import *

parser = argparse.ArgumentParser(description='My test of GraphSAGE')

parser.add_argument('--dataSet', type=str, default='phos-data')
parser.add_argument('--agg_func', type=str, default='MEAN')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--b_sz', type=int, default=20)
parser.add_argument('--seed', type=int, default=824)
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gcn', action='store_true')
parser.add_argument('--learn_method', type=str, default='sup')
parser.add_argument('--unsup_loss', type=str, default='normal')
parser.add_argument('--max_vali_f1', type=float, default=0)
parser.add_argument('--name', type=str, default='train')
parser.add_argument('--config', type=str,
                    default='/home/data/b532wangzeyu/MyGraphSageTest/exper_config_test.conf')
args = parser.parse_args()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        device_id = torch.cuda.current_device()
        # device_id = 1
        print('using device', device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cuda" if args.cuda else "cpu")
print('DEVICE:', device)


def evaluate(id_name, test_nodes_set, label_list_c, graphSage, classification, cnn, device, features, aa_adj_dict, features_protbert, seq_len):
    test_nodes = test_nodes_set
    labels = label_list_c

    # test_nodes_pos = []
    # for la_i, label in enumerate(labels):
    #     if label == 1:
    #         test_nodes_pos.append(la_i + 1)
    #     else:
    #         continue
    # nodes_pos_set = set(test_nodes_pos)
    # test_nodes_set = set(test_nodes)
    # nodes_neg_index = list(test_nodes_set - nodes_pos_set)
    # pos_num = list(labels).count(1)
    # if len(test_nodes) >= pos_num*2:
    #     test_nodes_neg = random.sample(nodes_neg_index, pos_num)
    #     test_nodes_ba = test_nodes_pos + test_nodes_neg
    # else:
    #     test_nodes_ba = test_nodes

    for model_i in models:
        model_i.eval()  # 将模型设置为评估模式
    with torch.no_grad():
        embs = graphSage(test_nodes, features, aa_adj_dict, seq_len)
        protbert_embs_batch = cnn(test_nodes, features_protbert)
        embs_concat = torch.cat([protbert_embs_batch, embs], dim=1)
        logists = classification(embs_concat)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)

        threshold = 0.5
        predicts = (logists > threshold).int()

        labels_test = []
        for test_node in test_nodes:
            labels_test.append(labels[test_node - 1])
        labels_test = torch.tensor(labels_test, dtype=torch.float)

        assert len(labels_test) == len(predicts)

        conf_matrix = confusion_matrix(labels_test, predicts.cpu())
        # print(conf_matrix)
        # 检查混淆矩阵是否包含零值
        # if np.any(conf_matrix):
        #     mcc = matthews_corrcoef(labels_test, predicts.cpu().data)
        # else:
        #     mcc = 0.0  # 或者你可以选择其他的处理方式

        with np.errstate(divide='ignore', invalid='ignore'):

            accuracy = accuracy_score(labels_test, predicts.cpu())
            accuracy = round(accuracy, 4)
            # print(id_name + " test_accuracy: " + str(accuracy))

            precision = precision_score(labels_test, predicts.cpu(), average="weighted")
            precision = round(precision, 4)
            # print(id_name + " test_precision: " + str(precision))

            recall = recall_score(labels_test, predicts.cpu(), average="weighted")
            recall = round(recall, 4)
            # print(id_name + " test_recall: " + str(recall))

            # mcc = matthews_corrcoef(labels_test, predicts.cpu().data)
            # mcc = round(mcc, 4)
            # print(id_name + " test_mcc: " + str(mcc))

            test_f1 = f1_score(labels_test, predicts.cpu(), average="weighted")
            test_f1 = round(test_f1, 4)
            # with open("/home/data/b532wangzeyu/MyGraphSageTest/test_result/test_F1_score", 'a+') as fileF1:
            #     fileF1.writelines(id_name + " test_f1: " + str(test_f1) + '\n')
            # print(id_name + " test_f1: " + str(test_f1))
            # 计算 ROC 曲线
            fpr, tpr, thresholds = roc_curve(labels_test, logists.cpu(), pos_label=1)
            # 计算 AUC
            roc_auc = auc(fpr, tpr)

    return accuracy, precision, recall, test_f1, roc_auc


if __name__ == '__main__':
    config = pyhocon.ConfigFactory.parse_file(args.config)

    test_id_list = test_data_load()

    test_nodes_num = len(test_id_list)
    acc_aver = 0
    pre_aver = 0
    recall_aver = 0
    f1_aver = 0
    # mcc_aver = 0
    roc_auc_aver = 0

    phar = tqdm(test_id_list, desc="Testing")
    for id in test_id_list:
        seq_len, seq, char_dict = char_map(id)
        feature_list, label_list = readContent(id)
        feature_protbert_list = read_protbert_feature(id, seq_len)
        aa_adj_dict = readCites(id)
        select_nodes_index, featureSTY_list, labelSTY_list = select_nodes(seq_len, char_dict, feature_list, label_list)

        features = torch.FloatTensor(feature_list).to(device)
        features_protbert = torch.FloatTensor(feature_protbert_list).to(device)
        model_pth = "/home/data/b532wangzeyu/MyGraphSageTest/STY_models_af_0305/model_best_train_ep0_0.8824.torch"
        # model_pth = "/home/data/b532wangzeyu/MyGraphSageTest/src_STY/STY_final_model.torch"

        models = torch.load(model_pth)
        [graphSage, classification, cnn] = models
        for model in models:
            model.to(device)

        acc_temp, pre_temp, recall_temp, f1_temp, roc_auc_temp = evaluate(id, select_nodes_index, label_list, graphSage,
                                                            classification, cnn, device, features, aa_adj_dict, features_protbert, seq_len)

        acc_aver = acc_aver + acc_temp
        pre_aver = pre_aver + pre_temp
        recall_aver = recall_aver + recall_temp
        f1_aver = f1_aver + f1_temp
        roc_auc_aver = roc_auc_aver + roc_auc_temp

        phar.update(1)
    phar.close()

    print("independent test_accuracy: " + str(round(float(acc_aver/test_nodes_num), 4)))

    print("independent test_precision: " + str(round(float(pre_aver/test_nodes_num), 4)))

    # print("independent test_recall: " + str(round(float(recall_aver/test_nodes_num), 4)))

    print("independent test_f1: " + str(round(float(f1_aver/test_nodes_num), 4)))

    # print("independent roc_auc_aver: " + str(round(float(roc_auc_aver / test_nodes_num), 4)))

