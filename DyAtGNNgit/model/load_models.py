def load_model(model_config):
    if model_config['model_name'] == 'DyAtGNN':
        from .model import DyAtGNN
        model = DyAtGNN(
            device=model_config['device'],
            node_feature=model_config['in_channels'],
            num_conv_layer=model_config['num_conv_layer'],
            num_of_nodes=model_config['num_of_nodes'],
            node_hidden=model_config['node_hidden'],
            feat_drop=model_config['feat_drop'],
            lamda=model_config['lamda'],
            alpha=model_config['alpha'],
            adaptive=2,
            attention_drop=model_config['attention_drop'],
            gcn_residual=model_config['use_residual'],
        )
    elif model_config['model_name'] == 'EvolveGCNO':
        from torch_geometric_temporal.nn.recurrent import EvolveGCNO
        model = EvolveGCNO(in_channels=model_config['in_channels']
                           )
    elif model_config['model_name'] == 'EvolveGCNH':
        from torch_geometric_temporal.nn.recurrent import EvolveGCNH
        model = EvolveGCNH(num_of_nodes=model_config['num_of_nodes'],
                           in_channels=model_config['in_channels']
                           )
    elif model_config['model_name'] == 'LRGCN':
        from torch_geometric_temporal.nn.recurrent import LRGCN
        model = LRGCN(in_channels=model_config['in_channels'],
                      out_channels=model_config['out_channels'],
                      num_relations=model_config['num_relations'],
                      num_bases=model_config['num_bases']
                      )
    elif model_config['model_name'] == 'GCLSTM':
        from torch_geometric_temporal.nn.recurrent import GCLSTM
        model = GCLSTM(in_channels=model_config['in_channels'],
                       out_channels=model_config['out_channels'],
                       K=model_config['K'],
                       normalization=model_config['normalization'],
                       # bias=model_config.bias,
                       )
    else:
        raise NotImplementedError()
    return model
