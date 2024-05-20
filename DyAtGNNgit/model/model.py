import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch_sparse as tspe
from torch.nn import GRU
from torch_geometric.nn.inits import glorot
from DyAtGNN.utils.utilities import *


###############################################################################
###############################################################################
###############################################################################

class GraphConvolutionNetworkII(nn.Module):

    def __init__(self, residual=False):
        super(GraphConvolutionNetworkII, self).__init__()
        self.residual = residual

    def forward(self, input, adj, h0, W, lamda, alpha, l):
        """here, h_0 and input has been already encoded to dense format"""
        theta = math.log(lamda / l + 1)
        # Tensor layout
        if adj.layout == torch.strided:
            hi = torch.matmul(adj, input)
        else:
            hi = tspe.spmm(adj._indices(), adj._values(), adj.size(0), adj.size(1), input)

        # GAT when adaptive = 1
        # output = torch.matmul(hi, W)

        support = (1 - alpha) * hi + alpha * h0
        output = theta * torch.matmul(support, W) + (1 - theta) * support
        if self.residual:
            output = output + input
        return output

###############################################################################
###############################################################################
###############################################################################

class DyAtGNN(nn.Module):
    def __init__(self, node_feature, num_conv_layer, num_of_nodes, node_hidden, feat_drop, lamda, alpha, device,
                 adaptive=2,
                 attention_drop=0.0, gcn_residual=False):
        super(DyAtGNN, self).__init__()
        self.device = torch.device(device)
        self.node_feature = node_feature
        self.num_conv_layer = num_conv_layer
        self.num_all_nodes = num_of_nodes
        # self.gcn_variant = gcn_variant
        self.gcn_residual = gcn_residual
        self.node_hidden = node_hidden
        self.activation = nn.ReLU()
        self.dropout = feat_drop
        self.alpha = alpha
        self.lamda = lamda
        self.adaptive = adaptive
        self.attention_drop = attention_drop
        self.weight = None
        self._create_layers()
        self._set_parameters()

    def _create_layers(self):

        self.recurrent_layer = GRU(
            input_size=self.node_hidden, hidden_size=self.node_hidden, num_layers=1
        )

        self.convs = nn.ModuleList()
        for _ in range(self.num_conv_layer):
            self.convs.append(GraphConvolutionNetworkII(residual=self.gcn_residual))

        self.hidden_layer = nn.Linear(self.node_feature, self.node_hidden)

        # If the feature dimension is too large or too small
        self.self_attention_layers = nn.ModuleList()
        self.self_attention_layers.append(nn.Linear(self.node_hidden, self.node_hidden))

        if self.adaptive == 2:
            self.self_attention_layers.append(nn.Linear(self.node_hidden, self.node_hidden))

    def _set_parameters(self):

        self.initial_weight = Parameter(torch.Tensor(self.node_hidden, self.node_hidden))
        glorot(self.initial_weight)

        self.self_attention_params = list(self.self_attention_layers.parameters())

        # add dynamic self-attention learning parameters
        if self.adaptive == 1:
            self.a_l = nn.Parameter(torch.zeros(size=(1, self.node_hidden)))
            self.a_r = nn.Parameter(torch.zeros(size=(1, self.node_hidden)))
            nn.init.xavier_uniform_(self.a_l)
            nn.init.xavier_uniform_(self.a_r)
        elif self.adaptive == 2:
            self.a = nn.Parameter(torch.zeros(size=(1, self.node_hidden)))
            nn.init.xavier_uniform_(self.a)

    # GAT
    def _dynamic_self_attention_v1(self, edge_index, a_l, a_r, hidden_feats):
        """
        Arg types:
            edge_index: [2, E] - edge of index.
            a_l,a_r: [1, D] - self_attention weight vector.
            hidden_feats: [N, D] - nodes feature matrix.
        Return types:
            A_trans: [N, N] - torch sparse matrix.
        """

        # attention
        alpha_l = (hidden_feats * a_l).sum(dim=-1).squeeze()
        alpha_r = (hidden_feats * a_r).sum(dim=-1).squeeze()
        alpha_l = alpha_l.index_select(0, edge_index[0])
        alpha_r = alpha_r.index_select(0, edge_index[1])

        edge_e = F.leaky_relu(alpha_l + alpha_r, 0.2)
        edge_e = edge_e - edge_e.max()
        edge_e = edge_e.exp()  # [E]

        edge_e = F.dropout(edge_e, p=self.attention_drop, training=self.training)

        assert not torch.isnan(edge_e).any()

        A_size = torch.Size([hidden_feats.size(0), hidden_feats.size(0)])
        # prepare and compute rowsum for softmax
        edge_e_sp = torch.sparse.FloatTensor(
            edge_index,
            edge_e,
            A_size
        )  # logical [TN, TN]

        e_rowsum = torch.sparse.sum(edge_e_sp, dim=1).to_dense()  # [TN]

        # generate div dense vector
        div_tensor = torch.index_select(e_rowsum, 0, edge_index[0])
        assert not torch.isnan(div_tensor).any()

        # softmax divide
        edge_e_sp_normalized = torch.sparse.FloatTensor(
            edge_index,
            edge_e.div(div_tensor + 1e-16),
            A_size
        )

        A_trans = edge_e_sp_normalized.t()
        assert not torch.isnan(A_trans).any()

        return A_trans

    # GATv2
    def _dynamic_self_attention_v2(self, edge_index, a, h_l, h_r):
        """
        Arg types:
            edge_index: [2, E] - edge of index.
            a: [1, D] - self_attention weight vector.
            h_l, h_r: [N, D] - nodes feature matrix.
        Return types:
            A_trans: [N, N] - torch sparse matrix.
        """

        alpha_l = (F.leaky_relu(h_l, 0.2) * a).sum(dim=-1).squeeze()
        alpha_r = (F.leaky_relu(h_r, 0.2) * a).sum(dim=-1).squeeze()  # [N]
        alpha_l = alpha_l.index_select(0, edge_index[0])  # [E] Select by index
        alpha_r = alpha_r.index_select(0, edge_index[1])  # [E]

        edge_e = alpha_l + alpha_r  # [E]
        edge_e = edge_e - edge_e.max()
        edge_e = edge_e.exp()

        edge_e = F.dropout(edge_e, p=self.attention_drop, training=self.training)
        assert not torch.isnan(edge_e).any()

        A_size = torch.Size([h_l.size(0), h_l.size(0)])
        # prepare and compute row sum for softmax
        edge_e_sp = torch.sparse.FloatTensor(
            edge_index,
            edge_e,
            A_size
        )  # logical [N, N]
        e_rowsum = torch.sparse.sum(edge_e_sp, dim=1).to_dense()

        # generate div dense vector
        div_tensor = torch.index_select(e_rowsum, 0, edge_index[0])
        assert not torch.isnan(div_tensor).any()

        # softmax divide
        edge_e_sp_normalized = torch.sparse.FloatTensor(
            edge_index,
            edge_e.div(div_tensor + 1e-16),  # Prevent division by 0
            A_size
        )

        A_trans = edge_e_sp_normalized.t()  # transposition
        assert not torch.isnan(A_trans).any()

        return A_trans

    def _topKpooling(self, node_embs, node_id, scores):

        vals, topk_indices = scores.view(-1).topk(self.node_hidden)
        topK_nodes = node_id[topk_indices]
        topK_nodes_features = node_embs[topk_indices]

        return topK_nodes_features, topK_nodes

    # Choose a pooling strategy based on the data
    def _pooling(self, x, remain_x, remain_nodes, added_x, added_nodes, node_scores):
        num_remain_pooling = self.node_hidden
        if x.size(0) <= self.node_hidden:
            X_pooling = torch.zeros((self.node_hidden - x.size(0), self.node_hidden), device=self.device)
            X_pooling = torch.cat((x, X_pooling), 0)
            pooling_node = remain_nodes
            num_remain_pooling = remain_x.size(0)
        else:
            if remain_x.size(0) <= self.node_hidden:
                X_pooling = remain_x
                pooling_node = remain_nodes
                num_remain_pooling = remain_x.size(0)
                if remain_x.size(0) < self.node_hidden:
                    vals, topk_indices = added_nodes.view(-1).topk(self.node_hidden - remain_x.size(0))
                    added_x_topk = added_x[topk_indices]
                    X_pooling = torch.cat((X_pooling, added_x_topk), 0)
            else:
                # Select the highest k nodes from the remaining nodes according to the score
                X_pooling, pooling_node = self._topKpooling(remain_x, remain_nodes, node_scores)
        return X_pooling, pooling_node, num_remain_pooling

    def forward(self, x, pre_x, edge_index, remain_nodes_index, added_nodes_index, node_id, node_scores):
        """
        :x: [N,D] Unprocessed feature matrix
        :edge_index: [2,E] the index of edge (adjacent matrix)
        :remain_nodes_index: [N-Nd] The index of remaining nodes,when the deleted node have been deleted
        :added_nodes_index: [Na] The index of added nodes
        :node_id: [N] Each of the node_id corresponds to each row of x
        :scores: Score vector for topK pooling
        :return: [TN, F=out_dim] (X_rnn , pooling_node_id , node_id)
        """

        # Delete the feature of the deleted node
        # remain_x: [N-Nd,D] The remaining features, when the deleted node have been deleted
        global GCN_output
        x = self.activation(self.hidden_layer(x))

        remain_nodes = node_id[remain_nodes_index]
        remain_x = x[remain_nodes_index]
        node_scores_index = remain_nodes_index
        if pre_x is not None:
            pre_rnn, pre_pooling_node,pre_node_id = pre_x[0], pre_x[1], pre_x[2]
            pre_pooling_node_index = torch.argwhere(torch.isin(pre_pooling_node, remain_nodes))
            node_scores_index = torch.argwhere(torch.isin(pre_node_id, remain_nodes))
            if pre_pooling_node_index.numel() != 0:
                remain_nodes_node_index = torch.argwhere(torch.isin(remain_nodes, pre_pooling_node))
                remain_x[remain_nodes_node_index] = pre_rnn[pre_pooling_node_index]

        added_x = x[added_nodes_index]

        added_nodes = node_id[added_nodes_index]
        node_scores = node_scores[node_scores_index]

        # H0 in GCNII
        X_residual = torch.zeros((self.num_all_nodes, self.node_hidden), device=self.device)
        X_residual[remain_nodes] = remain_x
        X_residual[added_nodes] = added_x

        # Choose a pooling strategy based on the data
        X_pooling, pooling_node, num_remain_pooling = self._pooling(x, remain_x, remain_nodes, added_x, added_nodes,
                                                                    node_scores)
        # to node hidden features

        # The feature and weight matrix input the RNN
        X_pooling = X_pooling[None, :, :]
        if self.weight is None:
            self.weight = self.initial_weight.data
        W = self.weight[None, :, :]
        X_rnn, W = self.recurrent_layer(X_pooling, W)

        # update node feature by rnn output, prepare for GCN
        X_GCN = X_residual
        X_GCN[pooling_node] = X_rnn.squeeze(dim=0)[0:num_remain_pooling, :]

        # DyAtGCN
        # self-attention module
        Wl_X = self.activation(self.self_attention_layers[0](X_GCN))
        if self.adaptive == 2:
            Wr_X = self.activation(self.self_attention_layers[-1](X_GCN))

        if self.adaptive == 1:
            adj = self._dynamic_self_attention_v1(edge_index, self.a_l, self.a_r, Wl_X)
        elif self.adaptive == 2:
            adj = self._dynamic_self_attention_v2(edge_index, self.a, Wl_X, Wr_X)

        # GCNII
        GCN_output = X_GCN
        for i, con in enumerate(self.convs):
            GCN_output = self.activation(
                con(GCN_output, adj, X_residual, W.squeeze(dim=0), self.lamda, self.alpha, i + 1))

        GCN_output = F.dropout(GCN_output, self.dropout, training=self.training)  # [N, F=out_dim]

        return GCN_output, (X_rnn.squeeze(dim=0).detach(), pooling_node.detach(), node_id.detach())

