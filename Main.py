import argparse
import csv
import random
from tqdm import tqdm
import pyhocon
import torch

from src_STY.DataCenter import *
from src_STY.DataCenter import _split_data
from src_STY.Models import *
from src_STY.utils import *
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='My test of GraphSAGE')

parser.add_argument('--dataSet', type=str, default='phos-data')
parser.add_argument('--agg_func', type=str, default='MEAN')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--b_sz', type=int, default=10)
parser.add_argument('--seed', type=int, default=824)
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gcn', action='store_true')
parser.add_argument('--learn_method', type=str, default='sup')
parser.add_argument('--unsup_loss', type=str, default='normal')
parser.add_argument('--max_vali_f1', type=float, default=0)
parser.add_argument('--name', type=str, default='train')
parser.add_argument('--config', type=str, default='/home/data/b532wangzeyu/MyGraphSageTest/exper_config_test.conf')
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

if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    config = pyhocon.ConfigFactory.parse_file(args.config)

    train_id_list = data_load()

    patience = 5
    current_patience = 0
    best_val_loss = 0.3

    for epoch in range(args.epochs):
        loss = 0.0
        print('----------------------EPOCH %d-----------------------' % epoch)
        # random.shuffle(train_id_list)

        phar = tqdm(train_id_list, desc="Training")
        for id in train_id_list:
            seq_len, seq, char_dict = char_map(id)
            feature_list, label_list = readContent(id)
            feature_protbert_list = read_protbert_feature(id, seq_len)
            aa_adj_dict = readCites(id)
            select_nodes_index, featureSTY_list, labelSTY_list = select_nodes(seq_len, char_dict, feature_list,
                                                                              label_list)
            features = torch.FloatTensor(feature_list).to(device)
            # print(id)
            features_protbert = torch.FloatTensor(feature_protbert_list).to(device)

            # 分割数据集
            test_indexs_list, val_indexs_list, train_indexs_list = _split_data(select_nodes_index, labelSTY_list,
                                                                               featureSTY_list.shape[0])

            graphSage = GraphSage(config['setting.num_layers'], features.size(1), 20,
                                  device, gcn=args.gcn, agg_func=args.agg_func)
            graphSage.to(device)

            num_labels = len(set(label_list))
            classification = Classification(40, 16, 1)
            classification.to(device)

            cnn = CNNLayer()
            cnn.to(device)

            graphSage, classification, cnn = apply_model(train_indexs_list, seq_len, select_nodes_index, label_list, graphSage,
                                                         classification, cnn, args.b_sz, device, args.learn_method, id,
                                                         features, aa_adj_dict, features_protbert)

            args.max_vali_f1, loss_temp = evaluate(id, test_indexs_list, val_indexs_list, label_list, graphSage,
                                                   classification, cnn, device,
                                                   args.max_vali_f1, args.name, epoch, features, aa_adj_dict,
                                                   features_protbert, seq_len)
            loss = loss + loss_temp
            phar.update(1)
        val_loss = loss/len(train_id_list)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("current best val loss:" + str(best_val_loss))
            current_patience += 1
            if current_patience == patience:
                print(f"Early stopping at epoch {epoch}")
                break
        else:
            current_patience = 0
        phar.close()
        print("current loss:" + str(val_loss))
# CUDA_VISIBLE_DEVICES=2 python -m my_src.Main --epochs 20 --cuda --learn_method sup
