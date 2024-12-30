import random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F


class SageLayer(nn.Module):
    def __init__(self, input_size, out_size, gcn=False):
        super(SageLayer, self).__init__()

        self.input_size = input_size
        self.out_size = out_size
        self.gcn = gcn

        self.weight = nn.Parameter(torch.FloatTensor(out_size, 2 * self.input_size))
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, self_feats, aggregate_feats, neighs=None):
        combined = torch.cat([self_feats, aggregate_feats], dim=1)
        combined = F.relu(self.weight.mm(combined.t())).t()
        combined = nn.functional.normalize(combined, p=2., dim=-1)
        return combined


class GraphSage(nn.Module):
    def __init__(self, num_layers, input_size, out_size, device, gcn=False, agg_func='MEAN'):
        super(GraphSage, self).__init__()

        self.num_layers = num_layers  # 2
        self.input_size = input_size  # 57
        self.out_size = out_size  # 57
        # self.raw_features = raw_features
        # self.aa_adj_dict = aa_adj_dict
        self.device = device
        self.gcn = gcn
        self.agg_func = agg_func

        for index in range(1, num_layers + 1):
            layer_size = out_size if index != 1 else input_size
            setattr(self, 'sage_layer' + str(index), SageLayer(layer_size, out_size, gcn=self.gcn))

    def forward(self, nodes_batch, raw_features, aa_adj_dict, seq_len):
        get_neigh_nodes_list = nodes_batch
        layer_data_batch = [(nodes_batch,)]
        for i in range(self.num_layers):
            layer_i_samp_neighs, layer_i_nodes_dict, layer_i_nodes_list = self._get_unique_neighs_list(
                get_neigh_nodes_list, aa_adj_dict, seq_len)
            get_neigh_nodes_list = layer_i_nodes_list

            layer_data_batch.insert(0, (layer_i_nodes_list, layer_i_samp_neighs, layer_i_nodes_dict))

        assert len(layer_data_batch) == self.num_layers + 1

        pre_hidden_embs = raw_features
        for index in range(1, self.num_layers + 1):
            nb = layer_data_batch[index][0]
            pre_neighs = layer_data_batch[index - 1]
            aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)

            # print("**************** aggregate_feats  **************")
            # print(aggregate_feats)

            sage_layer = getattr(self, 'sage_layer' + str(index))
            if index > 1:
                nb = self._nodes_map(nb, pre_neighs)

            nodes_list = []
            for node_index in nb:
                node_in_list = int(node_index) - 1
                nodes_list.append(node_in_list)

            cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nodes_list], aggregate_feats=aggregate_feats)
            pre_hidden_embs = cur_hidden_embs

        return pre_hidden_embs

    def _get_unique_neighs_list(self, nodes_batch, aa_adj_dict, seq_len):
        num_sample = 3
        _set = set
        # 邻居采样
        neighs = []
        # to_neighs = [self.aa_adj_dict[str(node)] for node in nodes_batch]
        for node in nodes_batch:
            seq_neibor_left = str(int(node) - 1)
            seq_neibor_right = str(int(node) + 1)
            dict_key = str(node)
            if dict_key not in aa_adj_dict:
                aa_adj_dict[dict_key] = {dict_key}
            if int(seq_neibor_left) != 0:
                aa_adj_dict[dict_key].add(seq_neibor_left)
            if int(seq_neibor_right) <= seq_len:
                aa_adj_dict[dict_key].add(seq_neibor_right)
            neighs.append(aa_adj_dict[dict_key])
        # print("this is inside function _get_unique_neighs_list")
        # print("one batch neighs:")
        # print(neighs)

        _sample = random.sample
        samp_neighs = [_set(_sample(neigh, num_sample)) if len(neigh) >= num_sample else neigh for neigh in neighs]

        # print("***************** samp_neighs ********************")
        # print(samp_neighs)

        # samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        for node_index, samp_neigh in enumerate(samp_neighs):
            node_self = nodes_batch[node_index]
            node_self = str(node_self)
            samp_neigh.add(node_self)
        # print("***************** after add self samp_neighs ********************")
        # print(samp_neighs)

        all_involved_nodes_no_repeat_list = list(set().union(*samp_neighs))
        no_repeat_list_index = list(range(len(all_involved_nodes_no_repeat_list)))
        # print("***************** all_involved_nodes_no_repeat_list ********************")
        # print(all_involved_nodes_no_repeat_list)

        all_involved_nodes_dict = dict(list(zip(all_involved_nodes_no_repeat_list, no_repeat_list_index)))
        # print("***************** all_involved_nodes_dict ********************")
        # print(all_involved_nodes_dict)

        return samp_neighs, all_involved_nodes_dict, all_involved_nodes_no_repeat_list

    def aggregate(self, nodes, pre_hidden_embs, pre_neighs):
        all_nodes_list, samp_neighs, nodes_dict = pre_neighs

        assert len(nodes) == len(samp_neighs)

        indicator = [(str(nodes[i]) in samp_neighs[i]) for i in range(len(samp_neighs))]
        get_node_error = []
        # if False in indicator:
        #     get_node_error.append(indicator.index(False))
        #     samp_neighs[indicator.index(False)].add(str(nodes[indicator.index(False)]))
        assert (False not in indicator)

        samp_neighs = [(samp_neighs[i] - set([str(nodes[i])])) for i in range(len(samp_neighs))]
        for ii in range(len(nodes)):
            if len(samp_neighs[ii]) == 0:
                samp_neighs[ii].add(str(nodes[ii]))

        if len(pre_hidden_embs) == len(nodes_dict):
            embed_matrix = pre_hidden_embs
        else:
            # 37
            embed_matrix = pre_hidden_embs[0].reshape(1, 57)
            for emb_index in all_nodes_list:
                emb_index_in_tensor = int(emb_index) - 1
                embed_temp = pre_hidden_embs[emb_index_in_tensor].reshape(1, 57)
                embed_matrix = torch.cat([embed_matrix, embed_temp], dim=0)
            embed_matrix = embed_matrix[1:]
            # print(embed_matrix)

        mask = torch.zeros(len(samp_neighs), len(nodes_dict))

        column_indices = [nodes_dict[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]

        mask[row_indices, column_indices] = 1

        if self.agg_func == 'MEAN':
            num_neigh = mask.sum(1, keepdim=True)
            mask = mask.div(num_neigh).to(embed_matrix.device)
            aggregate_feats = mask.mm(embed_matrix)
        elif self.agg_func == 'MAX':
            indexs = [x.nonzero() for x in mask == 1]
            aggregate_feats = []
            for feat in [embed_matrix[x.squeeze()] for x in indexs]:
                if len(feat.size()) == 1:
                    aggregate_feats.append(feat.view(1, -1))
                else:
                    aggregate_feats.append(torch.max(feat, 0)[0].view(1, -1))
            aggregate_feats = torch.cat(aggregate_feats, 0)

        return aggregate_feats

    def _nodes_map(self, nodes, neighs):
        layer_nodes, samp_neighs, layer_nodes_dict = neighs
        assert len(samp_neighs) == len(nodes)
        index = [layer_nodes_dict[str(x)] for x in nodes]  # 记录将上一层的节点编号。
        return index


class Classification(nn.Module):
    def __init__(self, emb_size, hidden_dim, output_dim):
        super(Classification, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(emb_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)

    def forward(self, embeds):
        emdeds_out = self.layer(embeds)
        logists = self.sigmoid(emdeds_out)
        # logists = torch.log_softmax(self.layer(embeds), 1)
        return logists


class CNNLayer(nn.Module):
    def __init__(self):
        super(CNNLayer, self).__init__()

        # CNN layers for input2
        self.cnn1_input = nn.Conv1d(in_channels=1, out_channels=512, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.cnn2_input = nn.Conv1d(in_channels=512, out_channels=20, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(20)
        self.init_params()

    def init_params(self):
        for layer in [self.cnn1_input, self.cnn2_input]:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)

    def forward(self, nodes_batch, input):
        # CNN layers for input2
        nodes_index = []
        for item in nodes_batch:
            nodes_index.append(item - 1)

        select_batch = input[nodes_index]
        select_batch = select_batch.unsqueeze(1)

        protbert_cnn_out1 = self.cnn1_input(select_batch)
        protbert_cnn_out1 = self.batch_norm1(protbert_cnn_out1)
        protbert_cnn_out1 = F.relu(protbert_cnn_out1)

        protbert_cnn_out2 = self.cnn2_input(protbert_cnn_out1)
        protbert_cnn_out2 = self.batch_norm2(protbert_cnn_out2)
        protbert_cnn_out2 = F.relu(protbert_cnn_out2)

        # 全局平均池化
        protbert_cnn_out = F.adaptive_avg_pool1d(protbert_cnn_out2, 1)  # 池化到输出大小为 (1, 32)
        # 调整形状
        protbert_cnn_out = protbert_cnn_out.view(protbert_cnn_out.size(0), -1)
        return protbert_cnn_out
