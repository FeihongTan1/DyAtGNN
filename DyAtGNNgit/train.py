import argparse
import logging
from os.path import join

from datetime import datetime

import time

import numpy as np
from pydgn.experiment.util import s2c
from torch.utils.data import DataLoader

from graphesn import DynamicGraphReservoir, initializer

from DyAtGNN.model.discrete_model import LRGCNModel, GCLSTMModel, EvolveGCN_H_Model, EvolveGCN_O_Model
from model.load_models import load_model
from DyAtGNN.utils.utilities import *

# How to run this experiments:
# python3 -u dyngesn.py --data $data --units 32 16 8 --sigma 0.9 0.5 0.1 --leakage 0.9 0.5 0.1 --lr 0.01 0.001 0.0001 --wd 0.001 0.0001 --batch $batch

elliptic = 'elliptic'
twitter = 'twitter'
as_733 = 'as_733'
bitcoin_alpha = 'bitcoin_alpha'
reddit = 'reddit'
wikipedia = 'wikipedia'

def set_logging(args):
    output_dir = create_save_path(args)
    log_file_path = output_dir + args.model_id + '.log'
    log_level = logging.INFO
    logging.basicConfig(filename=log_file_path, level=log_level, format='%(asctime)s - %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')
    logging.info(vars(args))


def eval(model, readout, X, y, datalist, T_range_list, criterion, device, dataset_name):
    if model != 'DynGESN':
        model.eval()
    readout.eval()
    y_true, y_conf = [], []
    outputlist = []
    with torch.no_grad():
        for T_range in T_range_list:
            if T_range.__len__() == 0: break
            node_scores = readout.inital_scores(datalist[T_range[0]].x.to(device))
            prev_h = None
            for t in T_range:
                snapshot = datalist[t]
                snapshot.to(device)

                if type(model).__name__ == 'DyAtGNN':
                    outputs, embs = model(snapshot.x, prev_h, snapshot.edge_index, snapshot.remain_nodes_index,
                                          snapshot.added_nodes_index, snapshot.node_id, node_scores)
                    prev_h = embs
                    out, node_scores = readout(outputs, snapshot)
                elif model == 'DynGESN':
                    out, _ = readout(X[t].to(device), snapshot.to(device))
                else:
                    out, embs = model.forward(snapshot, prev_h)
                    prev_h = embs.data

                y_conf.append(out.squeeze().cpu().detach())
                y_true.append(y[t].squeeze().float().cpu().detach())
                t += 1

    y_conf = torch.cat(y_conf)
    y_true = torch.cat(y_true)
    # Compute loss
    loss = criterion(y_conf, y_true).cpu().item()

    # Compute metrics
    if dataset_name != twitter:
        y_conf = torch.sigmoid(y_conf)
        y_pred = (y_conf > 0.5).float()
    else:
        y_pred = y_conf
    score = compute_metrics(dataset_name, y_true, y_pred, y_conf)

    return loss, score


class LinearReadout(torch.nn.Module):
    def __init__(self, num_features, num_initial_feature, num_targets, drop=0.3) -> None:
        super().__init__()
        self.link_readout = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(num_features * 2, num_targets, bias=True)
        )
        self.node_readout = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(num_features,
                            num_targets)
        )
        self.node_scores = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(num_targets, 1),
            torch.nn.ReLU()
        )
        self.inital_node_scores = torch.nn.Sequential(
            torch.nn.Linear(num_initial_feature, 1),
            torch.nn.ReLU()
        )

    def inital_scores(self, x):
        return self.inital_node_scores(x)

    def obtain_scores(self, x):
        if x.size(-1) < 2:
            return self.node_scores(x)
        result = self.node_readout(x)
        node_scores = self.node_scores(result)
        return node_scores

    def forward(self, X, snapshot):
        if 'link_pred_ids' in snapshot:
            source, target = snapshot.link_pred_ids
            x = torch.cat((X[source], X[target]), dim=-1)
            result = self.link_readout(x)
            node_scores = self.obtain_scores(X)
        else:
            x = X[snapshot.node_id]
            result = self.node_readout(x)
            node_scores = self.node_scores(result)
        return result, node_scores


def train_eval(batch_size, path, alpha, units, sigma, leakage, learning_rate, weight_decay, dataset_name,
               num_trials, device, num_epochs, model_config, CHECKPOINT_VALIDATION_FREQ):
    datalist, x_list, node_list, edge_index_list, _ = prepare_data(path, False)

    times_step = batch_size
    train_proportion = 0.70
    val_proportion = 0.85
    T_train, T_valid = int(len(edge_index_list) * 0.70), int(len(edge_index_list) * 0.85)
    train_data_range, val_data_range, test_data_range = data_split(times_step, len(datalist), train_proportion,
                                                                   val_proportion)
    y = [d.y for d in datalist]

    num_all_nodes = x_list[0].size(0)
    num_initial_feature = datalist[0].x.shape[-1]

    results = {
        'alpha': alpha,
        'embedding dim': units,
        'batch_size': batch_size,
        'lr': learning_rate,
        'wd': weight_decay
    }
    if dataset_name == twitter:
        train_mae, val_mae, test_mae, train_mse, val_mse, test_mse = [], [], [], [], [], []
    else:
        train_auroc, val_auroc, test_auroc = [], [], []
        train_f1, val_f1, test_f1 = [], [], []
        train_acc, val_acc, test_acc = [], [], []
        train_balanced_acc, val_balanced_acc, test_balanced_acc = [], [], []

        val_MAP, test_MAP = [], []
        train_loss, val_loss, test_loss = [], [], []

    for trial_index in range(num_trials):
        # Set the seed for the new trial
        set_all_seed(trial_index)
        # elliptic
        # set_all_seed(3 * (trial_index + 1))

        X = None
        model_config = update_model_config(model_confg, num_initial_feature, num_all_nodes)
        global model
        if model_config['model_name'] == 'DyAtGNN':
            x_selected, remain_nodes_index, added_nodes_index = select_graph_nodes(x_list, node_list)
            for x, r, a, d in zip(x_selected, remain_nodes_index, added_nodes_index, datalist):
                d.x = x
                d.update({'remain_nodes_index': r})
                d.update({'added_nodes_index': a})
            # mean_num_nodes,mean_num_deleted_nodes, mean_num_added_nodes = statistics_graph_nodes_variation(datalist)
            # initialize model
            model = load_model(model_config).to(device)
        elif model_config['model_name'] == 'DynGESN':
            # DynGESN
            reservoir = DynamicGraphReservoir(num_layers=1, in_features=num_initial_feature, hidden_features=units,
                                              return_sequences=True)
            reservoir.initialize_parameters(recurrent=initializer('uniform', sigma=sigma / alpha),
                                            input=initializer('uniform', scale=1),
                                            leakage=leakage)
            X = reservoir(edge_index=edge_index_list, input=x_list)
            model = 'DynGESN'
            X.to(device)
        else:
            if model_config['model_name'] == 'EvolveGCNH':
                model_class = EvolveGCN_H_Model
            elif model_config['model_name'] == 'EvolveGCNO':
                model_class = EvolveGCN_O_Model
            elif model_config['model_name'] == 'LRGCN':
                model_class = LRGCNModel
            elif model_config['model_name'] == 'GCLSTM':
                model_class = GCLSTMModel
            else:
                raise NotImplementedError()

            if 'link_pred_ids' in datalist[0]:
                readout_type = 'model.predictor.LinearLinkPredictor'
            else:
                readout_type = 'model.predictor.LinearNodePredictor'
            readout_class = s2c(readout_type)
            model = model_class(model_config['in_channels'], model_config['in_channels'], datalist[0].y.shape[-1],
                                readout_class,
                                model_config).to(device)
        logging.info(model)

        train_data_list = [datalist[i] for range_train in train_data_range for i in range_train]
        train_batch_size = int(batch_size * train_proportion)
        collate_fn = lambda samples_list: samples_list
        tr_loader = DataLoader(train_data_list, batch_size=train_batch_size, collate_fn=collate_fn, shuffle=False)

        if dataset_name == twitter:
            train_mae.append(np.inf)
            val_mae.append(np.inf)
            test_mae.append(np.inf)
            train_mse.append(np.inf)
            val_mse.append(np.inf)
            test_mse.append(np.inf)
        else:
            train_auroc.append(-np.inf)
            val_auroc.append(-np.inf)
            test_auroc.append(-np.inf)

            train_f1.append(-np.inf)
            val_f1.append(-np.inf)
            test_f1.append(-np.inf)

            val_MAP.append(-np.inf)
            test_MAP.append(-np.inf)

            train_acc.append(-np.inf)
            val_acc.append(-np.inf)
            test_acc.append(-np.inf)

            train_balanced_acc.append(-np.inf)
            val_balanced_acc.append(-np.inf)
            test_balanced_acc.append(-np.inf)

            train_loss.append(np.inf)
            val_loss.append(np.inf)
            test_loss.append(np.inf)
        # num_targets= datalist[0].y.shape[-1] units
        readout = LinearReadout(num_initial_feature=num_initial_feature, num_features=units,
                                num_targets=datalist[0].y.shape[-1]).to(device)
        # train sample imbalance
        if dataset_name == elliptic or dataset_name == reddit:
            pos_weight = torch.tensor([10.0])
        elif dataset_name == wikipedia:
            pos_weight = torch.tensor([8.3])
        else:
            pos_weight = None
        criterion = torch.nn.L1Loss() if dataset_name == twitter else torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if model_config['model_name'] == 'DynGESN':
            optimizer = torch.optim.AdamW(readout.parameters(), lr=learning_rate,
                                          weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(list(model.parameters()) + list(readout.parameters()), lr=learning_rate,
                                          weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        best_score = np.inf if dataset_name == twitter else 0.

        best_epoch = 0
        total_time = 0.0
        for epoch in range(num_epochs):
            if model != 'DynGESN':
                model.train()
            readout.train()
            batch_times = []
            t = time.time()
            batch_start_t = 0
            DynGESN_t = 0
            for batch in tr_loader:
                batch_time = time.time()
                prev_h = None
                node_scores = readout.inital_scores(x_list[batch_start_t].to(device))
                y_pred, y_true = [], []
                for snapshot in batch:
                    snapshot.to(device)

                    if model_config['model_name'] == 'DyAtGNN':
                        outputs, embs = model(snapshot.x, prev_h, snapshot.edge_index, snapshot.remain_nodes_index,
                                              snapshot.added_nodes_index, snapshot.node_id, node_scores)
                        prev_h = embs
                        out, node_scores = readout(outputs, snapshot)
                    elif model_config['model_name'] == 'DynGESN':
                        out, _ = readout(X[DynGESN_t].to(device), snapshot.to(device))
                    else:
                        out, embs = model.forward(snapshot, prev_h)
                        prev_h = embs.data

                    y_pred.append(out.squeeze().cpu())
                    y_true.append(snapshot.y.squeeze().cpu())
                    DynGESN_t += 1

                batch_start_t += batch_size
                # Perform a backward pass to calculate the gradients
                y_pred = torch.cat(y_pred)
                y_true = torch.cat(y_true).float()
                loss = criterion(y_pred, y_true)
                optimizer.zero_grad()
                loss.backward()
                # Update parameters
                optimizer.step()
                # Update the learning rate of the optimizer
                scheduler.step(loss.data.item())
                # Statistical batch time
                # batch_times.append(time.time() - batch_time)

            # mean_batch_times = np.mean(batch_times)
            epoch_time = time.time() - t
            total_time += epoch_time

            if ((epoch + 1) % CHECKPOINT_VALIDATION_FREQ == 0):

                # val_data_list = [datalist[i] for range_val in val_data_range for i in range_val]
                # test_data_list = [datalist[i] for range_test in test_data_range for i in range_test]
                # eval
                loss_vl, score_vl = eval(model, readout, X, y, datalist, val_data_range, criterion,
                                         device,
                                         dataset_name)
                loss_ts, score_ts = eval(model, readout, X, y, datalist, test_data_range, criterion,
                                         device,
                                         dataset_name)

                if dataset_name == twitter:
                    if score_vl['mae'] < best_score:
                        best_score = score_vl['mae']
                        best_epoch = epoch
                        # train_mae[-1] = score_tr['mae']
                        # train_mse[-1] = score_tr['mse']
                        val_mae[-1] = score_vl['mae']
                        test_mae[-1] = score_ts['mae']
                        val_mse[-1] = score_vl['mse']
                        test_mse[-1] = score_ts['mse']

                else:
                    tmp_score = score_vl[
                        'f1'] if dataset_name == elliptic or dataset_name == reddit or dataset_name == wikipedia else \
                    score_vl['auroc']
                    if tmp_score > best_score:
                        best_score = tmp_score
                        best_epoch = epoch
                        # train_auroc[-1] = score_tr['auroc']
                        # train_f1[-1] = score_tr['f1']
                        # train_acc[-1] = score_tr['acc']
                        # train_balanced_acc[-1] = score_tr['balanced_acc']
                        # train_loss[-1] = loss_tr

                        val_auroc[-1] = score_vl['auroc']
                        test_auroc[-1] = score_ts['auroc']

                        val_f1[-1] = score_vl['f1']
                        test_f1[-1] = score_ts['f1']

                        val_MAP[-1] = score_vl['MAP']
                        test_MAP[-1] = score_ts['MAP']

                        val_acc[-1] = score_vl['acc']
                        test_acc[-1] = score_ts['acc']

                        val_balanced_acc[-1] = score_vl['balanced_acc']
                        test_balanced_acc[-1] = score_ts['balanced_acc']

                        val_loss[-1] = loss_vl
                        test_loss[-1] = loss_ts

                # print model statistics
                print_model_statistics(dataset_name, epoch, epoch_time, score_vl, score_ts)
                if dataset_name == twitter:
                    logging.info(
                        '''Ep: %d, Epoch time: %1.5f,
                        Valmae: %1.5f, Testmae: %1.5f,
                         Valmse: %1.5f, Testmse: %1.5f,''' %
                        (epoch, epoch_time, score_vl['mae'], score_ts['mae'],
                         score_vl['mse'], score_ts['mse'])
                    )
                else:
                    logging.info(
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
                if epoch - best_epoch > 50:
                    break

        print("Total Time for %1.5f epoch: %1.5f." % (num_epochs, total_time))
        logging.info("Total Time for %1.5f epoch: %1.5f." % (num_epochs, total_time))
    if dataset_name == twitter:
        results.update({
            # 'mean_train_mae': np.mean(train_mae),
            'mean_val_mae': np.mean(val_mae),
            'mean_test_mae': np.mean(test_mae),
            # 'mean_train_mse': np.mean(train_mse),
            'mean_val_mse': np.mean(val_mse),
            'mean_test_mse': np.mean(test_mse),
            # 'train_mae': train_mae,
            'val_mae': val_mae,
            'test_mae': test_mae,
            # 'train_mse': train_mse,
            'val_mse': val_mse,
            'test_mse': test_mse
        })
    else:
        results.update({
            'mean_train_auroc': np.mean(train_auroc),
            'mean_val_auroc': np.mean(val_auroc),
            'mean_test_auroc': np.mean(test_auroc),
            # 'mean_train_f1': np.mean(train_f1),
            'mean_val_f1': np.mean(val_f1),
            'mean_test_f1': np.mean(test_f1),
            # 'mean_train_acc': np.mean(train_acc),
            'mean_val_acc': np.mean(val_acc),
            'mean_test_acc': np.mean(test_acc),
            # 'mean_train_loss': np.mean(train_loss),
            'mean_val_loss': np.mean(val_loss),
            'mean_test_loss': np.mean(test_loss),
            # 'train_auroc': train_auroc,
            'val_MAP': val_MAP,
            'test_MAP': test_MAP,
            'val_auroc': val_auroc,
            'test_auroc': test_auroc,
            # 'train_f1': train_f1,
            'val_f1': val_f1,
            'test_f1': test_f1,
            # 'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            # 'train_balanced_acc': train_balanced_acc,
            'val_balanced_acc': val_balanced_acc,
            'test_balanced_acc': test_balanced_acc,
            # 'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
        })
    return results


if __name__ == "__main__":
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='data name')
    parser.add_argument("--model", type=str, default='DyAtGNN', help="model name.")
    parser.add_argument("--model_id", type=str, default=str(datetime.now().strftime("%m_%d_%Y_%H_%M_%S")),
                        help="save mdl file path id.")
    parser.add_argument('--units', help='reservoir units per layer', type=int, default=32)  # nargs='+'
    parser.add_argument('--sigma', help='sigma for recurrent matrix initialization', type=float,
                        default=0.9)
    parser.add_argument('--leakage', help='leakage constant', type=float, default=0.9)
    parser.add_argument('--ld', help='readout lambda', type=float, default=1e-3)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.01)
    parser.add_argument('--wd', help='weight decay', type=float, default=0.001)
    parser.add_argument('--batch', help='batch size', type=int, default=8)
    parser.add_argument('--trials', help='number of trials', type=int, default=5)
    parser.add_argument('--device', help='device', type=str, default='cuda')
    parser.add_argument('--ridge', help='use ridge regression/classification as readout', action='store_true')
    parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs.")
    parser.add_argument("--valid_freq", type=int, default=1, help="validation logging frequency.")

    ### DyAtGNN setup ###
    parser.add_argument("--num_conv_layers", type=int, default=2, help="number of conv layers.")
    parser.add_argument("--num_hiddens", type=int, default=32, help="number of hidden units.")
    parser.add_argument("--feat_drop", type=float, default=1e-1, help="feature dropout percentage.")
    parser.add_argument("--lamda", type=float, default=0.5, help="lamda.")
    parser.add_argument("--alpha", type=float, default=0.1, help="alpha.")
    parser.add_argument("--attention_drop", type=float, default=1e-1, help="attention dropout percentage.")
    parser.add_argument("--use_residual", type=bool, default=False, help="whether use residual connecton.")

    args = parser.parse_args()

    assert args.dataset in [as_733, bitcoin_alpha, elliptic, twitter, reddit, wikipedia]

    device = torch.device(args.device)

    processed_data_path = "data/processedData/"
    data_path = join(processed_data_path, args.dataset) + '.pt'

    # DyGESN
    if args.dataset == as_733:
        data_alpha = 38.01178806531002
    elif args.dataset == bitcoin_alpha:
        data_alpha = 1.532752722140408
    elif args.dataset == elliptic:
        data_alpha = 6.471870352526263
    elif args.dataset == wikipedia:
        data_alpha = 69.09077636862125
    elif args.dataset == reddit:
        data_alpha = 63.30796153604874
    elif args.dataset == twitter:
        data_alpha = 8.931630544074462
    else:
        _, _, _, _, data_alpha = prepare_data(data_path, True)
    print(f'alpha = {data_alpha}')

    model_confg = initial_model_config(args)

    ### set logging ###
    set_logging(args)

    train_result = train_eval(args.batch, data_path, data_alpha, args.units, args.sigma, args.leakage, args.lr, args.wd,
                              args.dataset,
                              args.trials, device, args.num_epochs, model_confg, args.valid_freq)
    print('result: ' + str(train_result))
    logging.info('result: ' + str(train_result))
