import json
import os

import numpy as np
import torch
from graphesn.util import compute_dynamic_graph_alpha
from sklearn.metrics import *



twitter = 'twitter'
elliptic = 'elliptic'
as_733 = 'as_733'
bitcoin_alpha = 'bitcoin_alpha'
reddit = 'reddit'
wikipedia = 'wikipedia'


def create_save_path(args):
    output_dir = "./outputs/{}/{}/{}/".format(args.model, args.dataset, args.model_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Save arguments to json and txt files
    with open(os.path.join(output_dir, 'flags.json'), 'w') as outfile:
        json.dump(vars(args), outfile)

    with open(os.path.join(output_dir, 'flags.txt'), 'w') as outfile:
        for k, v in vars(args).items():
            outfile.write("{}\t{}\n".format(k, v))

    return output_dir


def prepare_data(path, compute_alpha=True):
    datalist = torch.load(path)
    edge_index_list = [d.edge_index for d in datalist]
    x_list = [d.x for d in datalist]
    node_list = [d.node_id for d in datalist]
    alpha = compute_dynamic_graph_alpha(edge_index_list) if compute_alpha else None
    return datalist, x_list, node_list, edge_index_list, alpha


def data_split(time_step, data_len, train_proportion, val_proportion):
    time = 0
    train_data_range = []
    val_data_range = []
    test_data_range = []
    while time < data_len:
        if data_len - time < time_step:
            time_step = data_len - time
        train_data_range.append(range(time, int(time + time_step * train_proportion)))
        val_data_range.append(range(int(time + time_step * train_proportion), int(time + time_step * val_proportion)))
        test_data_range.append(range(int(time + time_step * val_proportion), time + time_step))
        time += time_step
    return train_data_range, val_data_range, test_data_range


def compute_metrics(dataset_name, y_true, y_pred, y_conf=None):
    if dataset_name == twitter:
        score = {
            'mae': torch.nn.functional.l1_loss(y_pred, y_true).cpu().item(),
            'mse': torch.nn.functional.mse_loss(y_pred, y_true).cpu().item()
        }
    else:
        pos_label = 1
        # if dataset_name == reddit or dataset_name == wikipedia:
        #     pos_label = 2
        score = {
            # 'MRR': label_ranking_average_precision_score(y_true, y_conf),
            'Precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'MAP': average_precision_score(y_true, y_conf, pos_label=pos_label),
            'auroc': roc_auc_score(y_true, y_conf),
            'f1': f1_score(y_true, y_pred, average='micro'),
            'acc': accuracy_score(y_true, y_pred),
            'balanced_acc': balanced_accuracy_score(y_true, y_pred)
        }
        # precision, recall, thresholds1 = precision_recall_curve(y_true, y_conf,pos_label=2)
        # fpr, tpr, thresholds = roc_curve(y_true, y_conf,pos_label=2)
    return score


def set_all_seed(seed):
    """set all possible random seeds"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def initial_model_config(args):
    model_confg = {'model_name': args.model, 'device': args.device}
    if args.model == 'DyAtGNN':
        model_confg.update({
            'num_conv_layer': args.num_conv_layers,
            'node_hidden': args.units,
            'feat_drop': args.feat_drop,
            'lamda': args.lamda,
            'alpha': args.alpha,
            'attention_drop': args.attention_drop,
            'use_residual': args.use_residual
        })
    elif args.model == 'LRGCN':
        model_confg.update({
            'out_channels': args.units,
            'num_relations': 1,
            'num_bases': 10
        })
        if args.dataset == 'twitter':
            model_confg.update({
                'num_relations': 42,
                'num_bases': None
            })
    elif args.model == 'GCLSTM':
        model_confg.update({
            'out_channels': args.units,
        })
        if args.dataset == 'bitcoin_alpha':
            model_confg.update({
                'normalization': None,
                'K': 3
            })
        elif args.dataset == 'as_733':
            model_confg.update({
                'normalization': 'sym',
                'K': 2
            })
        else:
            model_confg.update({
                'normalization': None,
                'K': 2
            })
    elif model_confg['model_name'] == 'EvolveGCNH' or model_confg['model_name'] == 'EvolveGCNO':
        if args.dataset == 'elliptic':
            model_confg.update({
                'normalize': True,
            })
        else:
            model_confg.update({
                'normalize': False,
            })
    return model_confg


def update_model_config(model_confg, num_initial_feature, num_of_nodes):
    model_confg.update({'in_channels': num_initial_feature})
    if model_confg['model_name'] == 'EvolveGCNH' or model_confg['model_name'] == 'DyAtGNN':
        model_confg.update({
            'num_of_nodes': num_of_nodes,
        })
    return model_confg


# select the remaining and added parts of x and nodes compare the prev x and nodes
def select_graph_nodes(x_list, node_list):
    x_new_list = []
    remain_nodes_index = []
    added_nodes_index = []
    nodes_prev = torch.tensor([])
    for x, nodes in zip(x_list, node_list):
        x_new_list.append(torch.index_select(x, dim=0, index=nodes))
        remain_nodes_index.append(torch.argwhere(torch.isin(nodes, nodes_prev)).squeeze())
        added_nodes_index.append(torch.argwhere(torch.isin(nodes, nodes_prev, invert=True)).squeeze())
        nodes_prev = nodes
    return x_new_list, remain_nodes_index, added_nodes_index


# statistics_graph_nodes_variation,obtain the average number of deleted nodes and added nodes
def statistics_graph_nodes_variation(datalist):
    num_deleted_nodes = []
    num_added_nodes = []
    num_nodes = []
    num_nodes.append(datalist[0].node_id.numel())
    for i in range(len(datalist) - 1):
        num_deleted_node = datalist[i].node_id.numel() - datalist[i + 1].remain_nodes_index.numel()
        num_added_node = datalist[i+1].added_nodes_index.numel()
        num_nodes.append(datalist[i+1].node_id.numel())
        num_deleted_nodes.append(num_deleted_node)
        num_added_nodes.append(num_added_node)
    return np.sum(num_nodes),np.sum(num_deleted_nodes),np.sum(num_added_nodes)


# select top k value
def topKselect(nodes, k):
    vals, topk_indices = nodes.view(-1).topk(k)  # select top k value
    topk_indices = topk_indices[vals > -float("Inf")]

    if topk_indices.size(0) < k:
        topk_indices = pad_with_last_val(topk_indices, k)  # Complete topk_indices

    return topk_indices


def pad_with_last_val(vect, k):
    device = 'cuda' if vect.is_cuda else 'cpu'
    pad = torch.ones(k - vect.size(0),
                     dtype=torch.long,
                     device=device) * vect[-1]
    vect = torch.cat([vect, pad])
    return vect


# cat two tensor by the index of data in tensor
def tensor_cat_by_index(a, b, a_index, b_index):
    result = torch.zeros((a.size(0) + b.size(0), a.size(1)))
    result[a_index] = a
    result[b_index] = b
    return result


def print_model_statistics(dataset_name, epoch, epoch_time, score_vl, score_ts):
    if dataset_name == twitter:
        print(
            '''Ep: %d, Epoch time: %1.5f,
                        Valmae: %1.5f, Testmae: %1.5f,
                         Valmse: %1.5f, Testmse: %1.5f,''' %
            (epoch, epoch_time, score_vl['mae'], score_ts['mae'],
             score_vl['mse'], score_ts['mse'])
        )
    else:
        print(
            '''Ep: %d, Epoch time: %1.5f,
                        ValMAP: %1.5f, TestMAP: %1.5f,
                         ValAcc: %1.5f, TestAcc: %1.5f,
                         ValBaAcc: %1.5f, TestBaAcc: %1.5f,
                         ValF1: %1.5f, TestF1: %1.5f,
                         ValAUROC: %1.5f, TestAUROC: %1.5f,''' %
            (epoch, epoch_time, score_vl['MAP'], score_ts['MAP'],
             # score_vl['MRR'], score_ts['MRR'],
             score_vl['acc'], score_ts['acc'],
             score_vl['balanced_acc'], score_ts['balanced_acc'],
             score_vl['f1'], score_ts['f1'],
             score_vl['auroc'], score_ts['auroc']
             )
        )

