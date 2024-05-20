import os

import kaggle
import torch

from pydgn.data.dataset import TemporalDatasetInterface
from torch_geometric.data import download_url, extract_tar, extract_gz, Data
from torch_geometric.datasets import JODIEDataset
from torch_geometric.utils import negative_sampling, degree
from os.path import join, isfile, isdir
import pandas as pd
import numpy as np
from torch_geometric_temporal import TwitterTennisDatasetLoader
from tqdm import tqdm
from datetime import datetime


# **** LINK PREDICTION  ****
class AutonomousSystemsDatasetInterface(TemporalDatasetInterface):
    """
    Autonomous systems AS-733 interface.
    It contains a sequece of graph snapshots for link prediction.
    Available at https://snap.stanford.edu/data/as-733.html
    """

    def __init__(self, original_data_path, processed_data_path, name='as_733'):
        self.original_data_path = join(original_data_path, name)
        self.processed_data_path = join(processed_data_path, name)
        self.name = name
        self._check_and_download()

        if not isfile(self.processed_data_path + '.pt'):
            self.dataset = self._load_data(self.original_data_path)
        else:
            self.dataset = torch.load(self.processed_data_path + '.pt')

    def _check_and_download(self):

        if not isdir(self.original_data_path):
            tar_file = download_url(f'https://snap.stanford.edu/data/as-733.tar.gz', self.root)
            extract_tar(tar_file, join(self.root, self.name), mode='r:gz')
            os.unlink(tar_file)

    def _load_data(self, path):
        data_list = []

        graph_paths = sorted([join(path, f) for f in os.listdir(path) if isfile(join(path, f))])

        # Map to continuous node id
        nodes_ids = set()
        for i in range(len(graph_paths)):
            g = pd.read_csv(graph_paths[i], skiprows=4, sep='\t', names=['from', 'to'])
            nodes_ids.update(g.values.flatten())

            # nodes_ids.update(g['to'].values)
        map_id = {old_id: i for i, old_id in enumerate(nodes_ids)}

        embed_size = 128
        # max_degree = 0
        # nodes_degree = set()
        num_all_nodes = len(map_id)  # By setting the maximum number of nodes, we consider them fixed along time
        g = None
        for i in tqdm(range(len(graph_paths) - 1)):
            if g is None:
                g = pd.read_csv(graph_paths[i], skiprows=4, sep='\t', names=['from', 'to'])
                g = g.applymap(lambda x: map_id[x])  # Map to continuous node id
                f = open(graph_paths[i], 'r')
                stats = f.readlines()[2]
                f.close()
                num_nodes = int(stats.split('\t')[0].split(':')[1])

            g_next = pd.read_csv(graph_paths[i + 1], skiprows=4, sep='\t', names=['from', 'to'])
            g_next = g_next.applymap(lambda x: map_id[x])  # Map to continuous node id

            # Extracts all unique values in the entire DataFrame and converts them to set
            node_set = set(g.values.flatten())
            node_id = torch.tensor(list(node_set), dtype=torch.int).contiguous()
            # node features set an all-1 tensor of dimension 128
            x = torch.ones((num_all_nodes, embed_size))
            edge_index = torch.from_numpy(g.to_numpy().T)
            # Set node features by degree
            # x_degree = degree(edge_index[0], num_all_nodes, dtype=torch.long)
            # max_degree = max(max_degree, x_degree.max().item())
            # nodes_degree.update(torch.unique(x_degree[node_id]).tolist())

            f = open(graph_paths[i + 1], 'r')
            stats_next = f.readlines()[2]
            f.close()
            num_nodes_next = int(stats_next.split('\t')[0].split(':')[1])
            edge_index_next = torch.from_numpy(g_next.to_numpy().T)

            # Negative sampling:
            # Samples random negative edges of a graph given by the graph at time i+1
            neg_edge_index = negative_sampling(edge_index=edge_index_next,
                                               num_nodes=num_nodes_next)

            neg_edge_index = torch.cat((neg_edge_index, torch.zeros(1, neg_edge_index.size(1))))
            edge_index_next = torch.cat((edge_index_next, torch.ones(1, edge_index_next.size(1))))
            target_edge_index = torch.cat((neg_edge_index, edge_index_next), 1)

            relation_type = torch.zeros(edge_index.shape[1])
            data_list.append(Data(x=x,
                                  node_id=node_id,
                                  edge_index=edge_index,
                                  y=target_edge_index[2].unsqueeze(-1).type(torch.FloatTensor),
                                  relation_type=relation_type.type(torch.LongTensor),
                                  link_pred_ids=target_edge_index[:-1].type(torch.LongTensor)))

            g = g_next
            stats = stats_next
            num_nodes = num_nodes_next

        # embed_layer = torch.nn.Linear(max_degree + 1, embed_size)
        # # one-hot node-degree and linear layer as the input feature
        # for i in range(len(data_list) - 1):
        #     degree_embeddings = torch.nn.functional.one_hot(data_list[i].node_degree,
        #                                                     num_classes=max_degree + 1).float()
        #     x = embed_layer(degree_embeddings)
        #     setattr(data_list[i], 'x', x)
        #     delattr(data_list[i], 'node_degree')
        #     del x, degree_embeddings

        torch.save(data_list, self.processed_data_path + '.pt')

        return data_list

    def __getitem__(self, idx):
        data = self.dataset[idx]
        setattr(data, 'mask', self.get_mask(data))
        return data

    def get_mask(self, data):
        # in this case data is a Data object containing a snapshot of a single
        # graph sequence.
        # the task is link prediction at each time step, so the mask is always true
        mask = np.ones((1, 1))  # time_steps x 1
        return mask

    @property
    def dim_node_features(self):
        return self.dataset[0].x.shape[-1]

    @property
    def dim_edge_features(self):
        return 1

    @property
    def dim_target(self):
        # binary link prediction
        return 1

    def __len__(self):
        return len(self.dataset)


class BitcoinAlphaDatasetInterface(TemporalDatasetInterface):
    """
    Bitcoin Alpha network interface for link prediction on discrete-time dynamic graphs.
    It contains a who-trusts-whom network of people who trade using Bitcoin on a platform 
    called Bitcoin Alpha.
    Available at https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html
    """

    def __init__(self, original_data_path, processed_data_path, name='bitcoin_alpha'):
        self.original_data_path = join(original_data_path, name)
        self.processed_data_path = join(processed_data_path, name)
        self.name = name
        self._check_and_download()

        if not isfile(self.processed_data_path + '.pt'):
            self.dataset = self._load_data(self.original_data_path)
        else:
            self.dataset = torch.load(self.processed_data_path + '.pt')

    def _check_and_download(self):
        if not isdir(self.original_data_path):
            gz_file = download_url(f'https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz',
                                   self.original_data_path)
            extract_gz(gz_file, self.original_data_path)
            os.unlink(gz_file)
            extracted_gz = join(self.original_data_path, 'soc-sign-bitcoinalpha.csv')
            self.original_data_file = join(self.original_data_path + '/' + self.name + '.csv')
            if not isfile(self.original_data_file):
                os.rename(extracted_gz, self.original_data_file)

    def _load_data(self, path):
        data_list = []

        df = pd.read_csv(path + '/' + self.name + '.csv', names=['SOURCE', 'TARGET', 'RATING', 'TIME'])

        # Convert the timestamp to year-month-day
        convert_date = lambda x: str(datetime.fromtimestamp(x)).split(' ')[0]
        df.TIME = pd.to_datetime(df.TIME.apply(convert_date))
        # Convert the timestamp to year-month
        df.TIME = df.TIME.dt.strftime('%Y-%m')
        # Map to continuous node id
        nodes_ids = set()
        nodes_ids.update(df['SOURCE'].values)
        nodes_ids.update(df['TARGET'].values)
        map_id = {old_id: i for i, old_id in enumerate(nodes_ids)}
        df[['SOURCE', 'TARGET']] = df[['SOURCE', 'TARGET']].applymap(lambda x: map_id[x])
        num_nodes = len(map_id)

        # Daily aggregation of snapshots 
        data_list = []
        max_degree = 0
        for _, g in df.groupby('TIME'):
            edge_index_numpy = g[['SOURCE', 'TARGET']].to_numpy().T
            edge_index = torch.from_numpy(edge_index_numpy)

            # Extracts all unique values in the entire DataFrame and converts them to set
            node_set = set(edge_index_numpy.flatten())
            node_id = torch.tensor(list(node_set), dtype=torch.int).contiguous()

            x_degree = degree(edge_index[0], num_nodes, dtype=torch.long)
            max_degree = max(max_degree, x_degree.max().item())
            data_list.append(Data(node_degree=x_degree[node_id], node_id=node_id, edge_index=edge_index))

        for i in tqdm(range(len(data_list) - 1)):
            edge_index = data_list[i].edge_index
            # one-hot node-degree as the input feature
            x = torch.zeros(num_nodes, max_degree + 1)
            x[data_list[i].node_id, data_list[i].node_degree] = 1
            # x = torch.nn.functional.one_hot(data_list[i].node_degree,num_classes=max_degree + 1)

            next_edge_index = data_list[i + 1].edge_index
            # Negative sampling:
            # Samples random negative edges of a graph given by the graph at time i+1
            neg_edge_index = negative_sampling(edge_index=next_edge_index,
                                               num_nodes=num_nodes)

            neg_edge_index = torch.cat((neg_edge_index, torch.zeros(1, neg_edge_index.size(1))))
            next_edge_index = torch.cat((next_edge_index,
                                         torch.ones(1, data_list[i + 1].edge_index.size(1))))
            target_edge_index = torch.cat((neg_edge_index, next_edge_index), 1).type(torch.LongTensor)

            relation_type = torch.zeros(edge_index.shape[1])
            setattr(data_list[i], 'x', x)
            setattr(data_list[i], 'relation_type', relation_type.type(torch.LongTensor))
            setattr(data_list[i], 'y', target_edge_index[2].unsqueeze(-1).type(torch.FloatTensor))
            setattr(data_list[i], 'link_pred_ids', target_edge_index[:-1])

        torch.save(data_list[:-1], self.processed_data_path + '.pt')

        return data_list

    def __getitem__(self, idx):
        data = self.dataset[idx]
        setattr(data, 'mask', self.get_mask(data))
        return data

    def get_mask(self, data):
        # in this case data is a Data object containing a snapshot of a single
        # graph sequence.
        # the task is link prediction at each time step, so the mask is always true
        mask = np.ones((1, 1))  # time_steps x 1
        return mask

    @property
    def dim_node_features(self):
        return self.dataset[0].x.shape[-1]

    @property
    def dim_edge_features(self):
        return 1

    @property
    def dim_target(self):
        # binary link prediction
        return 1

    def __len__(self):
        return len(self.dataset)


# **** NODE PREDICTION  ****
class TwitterTennisDatasetInterface(TemporalDatasetInterface):
    """
    Twitter Tennis Dataset.
    It contains Twitter mention graphs related to major tennis tournaments from 2017.
    Each snapshot change with respect to edges and features.
    """

    def __init__(self, root, name, event_id='rg17', num_nodes=1000,
                 feature_mode='encoded', target_offset=1):

        assert event_id in ['rg17', 'uo17'], f'event_id can be rg17 or uo17, not {event_id}'
        assert num_nodes <= 1000, f'num_nodes must be less or equal to 1000, not {num_nodes}'
        assert feature_mode in [None, 'diagonal',
                                'encoded'], f'feature_mode can be None, diagonal, or encoded. It can not be {feature_mode}'

        self.root = root
        self.name = name
        self.event_id = event_id
        self.num_nodes = num_nodes
        self.feature_mode = feature_mode
        self.target_offset = target_offset

        path = join(self.root, self.name) + '.pt'
        if not isfile(path):
            self.dataset = TwitterTennisDatasetLoader(
                event_id=self.event_id,
                N=self.num_nodes,
                feature_mode=self.feature_mode,
                target_offset=self.target_offset
            ).get_dataset()
            torch.save(self.data_preprocess(), path)
        else:
            self.dataset = torch.load(path)

    def data_preprocess(self):
        data_list = []
        for snapshot in self.dataset:
            node_list = snapshot.edge_index.reshape(-1).tolist()
            node_list.sort()
            node_set = set(node_list)
            node_id = torch.tensor(list(node_set), dtype=torch.int).contiguous()
            snapshot.y = (snapshot.y[node_id]).unsqueeze(-1)
            node_mask = torch.zeros(snapshot.x.shape[0])
            node_mask[node_id] = 1
            setattr(snapshot, 'node_id', node_id)
            setattr(snapshot, 'node_mask', node_mask.bool())
            setattr(snapshot, 'relation_type', (snapshot.edge_attr - 1).type(torch.LongTensor))
            data_list.append(snapshot)
        return data_list

    @property
    def dim_node_features(self):
        return self.dataset.features[0].shape[1]

    @property
    def dim_edge_features(self):
        return 1

    @property
    def dim_target(self):
        # node regression: each time step is a tuple
        return 1

    def get_mask(self, data):
        # in this case data is a Data object containing a snapshot of a single
        # graph sequence.
        # the task is node classification at each time step
        mask = np.ones((1, 1))  # time_steps x 1
        return mask

    def __len__(self):
        return len(self.dataset.features)

    def __getitem__(self, time_index):
        data = self.dataset.__getitem__(time_index)
        setattr(data, 'mask', self.get_mask(data))
        setattr(data, 'relation_type', data.edge_attr - 1)
        return data


class EllipticDatasetInterface(TemporalDatasetInterface):
    """
    Elliptic Dataset.
    The data maps Bitcoin transactions to real entities belonging to licit categories versus
    illicit ones.
    Each snapshot change with respect to nodes, edges, and features.
    """

    def __init__(self, original_data_path, processed_data_path, name='elliptic'):
        self.original_data_path = join(original_data_path, name)
        self.processed_data_path = join(processed_data_path, name)
        self.name = name
        self._check_and_download()

        if not isfile(self.processed_data_path + '.pt'):
            self.dataset = self._load_data(self.original_data_path)
        else:
            self.dataset = torch.load(self.processed_data_path + '.pt')

    def _check_and_download(self):
        if not isdir(self.original_data_path):
            print('Downloading data...')
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files('ellipticco/elliptic-data-set', path=self.original_data_path,
                                              unzip=True)  # <self.root>/<self.name>

    def _load_data(self, path):
        print('Loading data...')
        path_classes = join(path, 'elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
        path_edgelist = join(path, 'elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')
        path_features = join(path, 'elliptic_bitcoin_dataset/elliptic_txs_features.csv')

        classes = pd.read_csv(path_classes, index_col='txId')  # labels for the transactions i.e. 'unknown', '1', '2'
        edgelist = pd.read_csv(path_edgelist, index_col='txId1')  # directed edges between transactions
        features = pd.read_csv(path_features, header=None, index_col=0)  # features of the transactions

        # Select only the labeled transactions
        labelled_classes = classes[classes['class'] != 'unknown']
        labelled_tx = set(list(labelled_classes.index))

        # Map node_id in the range [0, num_nodes]
        nodes = features.index
        nodes_set = set(nodes.tolist())
        map_id = {j: i for i, j in enumerate(nodes_set)}  # mapping nodes to indexes

        map_class = {'1': 1, '2': 0}

        # Compute Data object for each timestep
        data_list = []
        for timestep, df_group in tqdm(features.groupby(1)):
            # Keep only labelled nodes
            labelled_nodes = sorted([tx for tx in df_group.index if tx in labelled_tx])
            df_group = df_group.loc[labelled_nodes]

            # Get edge_index associated to the current timestep
            edge_index = edgelist.loc[edgelist.index.intersection(
                labelled_nodes).unique()]  # Find the edge that intersects with the labelled node

            # We consider only edges and features as dynamic, i.e., nodes are fixed on the temporal axis
            # node_id map to each row of x
            node_id = torch.tensor(df_group.index.map(map_id)).contiguous()  # nodes mapped id
            x = torch.tensor(features[range(2, 167)].values, dtype=torch.float).contiguous()  # nodes features
            # map_id = {j:i for i,j in enumerate(labelled_nodes)} # node index, value mapping based on new snapshot
            edge_index['txId1'] = edge_index.index.map(map_id)  # Replace the number of the original node
            edge_index.txId2 = edge_index.txId2.map(map_id)

            node_mask = torch.zeros(x.shape[0])
            node_mask[[map_id[n] for n in labelled_nodes]] = 1

            edge_index = np.array(edge_index[['txId1', 'txId2']].values).T
            edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()

            # labelled node real class
            targets = [map_class[v] for v in classes.loc[labelled_nodes, 'class'].tolist()]
            y = torch.tensor(targets).contiguous()

            relation_type = torch.zeros(edge_index.shape[1])
            data = Data(
                node_id=node_id,
                edge_index=edge_index,
                x=x,
                y=y.unsqueeze(-1),  # labelled nodes
                relation_type=relation_type.type(torch.LongTensor),  # torch.zeros(edge_index.shape[1]),
                node_mask=node_mask.bool()
            )
            data_list.append(data)

        torch.save(data_list, self.processed_data_path + '.pt')
        return data_list

    def __getitem__(self, idx):
        data = self.dataset[idx]
        setattr(data, 'mask', self.get_mask(data))
        return data

    def get_mask(self, data):
        # We predict at each snapshot all the nodes
        mask = np.ones((1, 1))  # time_steps x 1
        return mask

    @property
    def dim_node_features(self):
        return self.dataset[0].x.shape[-1]

    @property
    def dim_edge_features(self):
        return 1

    @property
    def dim_target(self):
        # binary node classification
        return 1

    def __len__(self):
        return len(self.dataset)


class JODIEDatasetInterface(TemporalDatasetInterface):
    """
    JODIE data
    JODIE = ['Wikipedia', "Reddit", "LastFM"]
    Divide into n snapshots, Each snapshot change with respect to nodes, edges, and features.
    """

    def __init__(self, root, name='wikipedia'):
        self.root = root
        self.name = name
        self.Timestamps = 100
        self.JODIE = ['reddit', 'wikipedia', 'mooc', 'lastfm']

        path = join(self.root, self.name) + '.pt'
        if not isfile(path):
            torch.save(self.get_dataset(self.name), path)
        else:
            self.dataset = torch.load(path)

    def get_dataset(self, name):

        if name in self.JODIE:
            dataset = JODIEDataset(self.root, name)
            ctdg_data = dataset[0]
            edge_indices = self.chunk_tensor(ctdg_data.edge_index, self.Timestamps, 1)
            msg_list = self.chunk_tensor(ctdg_data.msg, self.Timestamps, 0)
            num_user = 10000
            if name == 'wikipedia':
                num_user = 8277

            data_list = []
            for edge_index, edge_feature in zip(edge_indices, msg_list):
                relation_type = torch.zeros(edge_index.shape[1])
                node_list = edge_index.reshape(-1).tolist()
                node_list.sort()
                node_set = set(node_list)
                node_id = torch.tensor(list(node_set), dtype=torch.int).contiguous()
                x = torch.ones((ctdg_data.num_nodes, 128), dtype=torch.float32)
                y_list = []
                for node in node_id:
                    if node >= num_user:
                        y_list.append(1)
                    else:
                        y_list.append(0)
                y = torch.tensor(y_list).contiguous()
                node_mask = torch.zeros(x.shape[0])
                node_mask[node_id] = 1
                data = Data(
                    node_id=node_id,
                    edge_index=edge_index,
                    x=x,
                    y=y.unsqueeze(-1),  # labelled nodes
                    relation_type=relation_type.type(torch.LongTensor),  # torch.zeros(edge_index.shape[1]),
                    node_mask=node_mask.bool()
                )
                data_list.append(data)
            return data_list
        else:
            raise NotImplementedError

    def chunk_tensor(self, tensor, n, dim=0):

        size_dim = tensor.size(dim)
        avg_length = size_dim // n
        remainder = size_dim % n

        chunks = []

        start_idx = 0
        for i in range(n):
            length = avg_length + 1 if i < remainder else avg_length

            chunk = tensor.narrow(dim, start_idx, length)
            chunks.append(chunk)

            start_idx += length
        return chunks

    @property
    def dim_target(self):
        return 1


if __name__ == "__main__":
    original_data_path = "originalData/"
    processed_data_path = "processedData/"

    # sb = BitcoinAlphaDatasetInterface(original_data_path, processed_data_path)
    # sb._load_data(join(original_data_path, 'bitcoin_alpha'))  # debug load_data

    # sa = AutonomousSystemsDatasetInterface(original_data_path, processed_data_path)
    # sa._load_data(join(original_data_path, 'as_733'))  # debug load_data

    # s = EllipticDatasetInterface(original_data_path,processed_data_path)
    # s._load_data(join(original_data_path, 'elliptic'))

    # st = TwitterTennisDatasetInterface(processed_data_path, 'twitter')

    sw = JODIEDatasetInterface(processed_data_path, 'wikipedia')

    sr = JODIEDatasetInterface(processed_data_path, 'reddit')
