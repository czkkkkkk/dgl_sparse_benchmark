from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CoraGraphDataset
import torch
import dgl

def OgbDataset(graph_name, dev):
    assert graph_name in ('ogbn-products', 'ogbn-arxiv')
    dataset = DglNodePropPredDataset(name=graph_name, root='/tmp')
    g, labels = dataset[0]
    g = dgl.add_self_loop(g).to(dev)
    split_idx = dataset.get_idx_split()
    # get split labels
    g.ndata['label'] = labels.to(dev).view(-1)
    g.ndata['train_mask'] = torch.zeros(g.number_of_nodes(), dtype=torch.bool, device=dev)
    g.ndata['train_mask'][split_idx['train']] = True
    g.ndata['val_mask'] = torch.zeros(g.number_of_nodes(), dtype=torch.bool, device=dev)
    g.ndata['val_mask'][split_idx['valid']] = True
    g.ndata['test_mask'] = torch.zeros(g.number_of_nodes(), dtype=torch.bool, device=dev)
    g.ndata['test_mask'][split_idx['test']] = True
    num_labels = len(
        torch.unique(labels[torch.logical_not(torch.isnan(labels))])
    )
    print('NumNodes: ', g.number_of_nodes())
    print('NumEdges: ', g.number_of_edges())
    print('NumFeats: ', g.ndata['feat'].shape[1])
    print('NumClasses: ', num_labels)
    print('NumTrainingSamples: ', split_idx['train'].shape[0])
    print('NumValidationSamples: ', split_idx['valid'].shape[0])
    print('NumTestSamples: ', split_idx['test'].shape[0])
    return g, num_labels

def CoraDataset(dev):
    dataset = CoraGraphDataset()
    g = dataset[0].to(dev)
    labels = g.ndata['label']
    num_labels = len(
        torch.unique(labels[torch.logical_not(torch.isnan(labels))])
    )
    return g, num_labels

def load_dataset(name, dev):
    if name == 'cora':
        return CoraDataset(dev)
    else:
        return OgbDataset(name, dev)