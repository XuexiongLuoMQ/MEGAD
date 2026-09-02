import os
import dgl
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import torch
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics import precision_score, recall_score, f1_score
#from pygod.utils import load_data
#from pygodm.pygod.utils import load_data

def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix.
    """
    adj = (adj + sp.eye(adj.shape[0]))
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    np.seterr(divide='ignore', invalid='ignore')
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo().toarray()


def modularity_matrix(adj):
    m = torch.sum(adj) / 2  # Total number of edges

    # Calculate the degree of each node
    degrees = torch.sum(adj, dim=1)

    # Calculate the expected number of edges between nodes i and j if edges are placed randomly
    expected_edges = degrees.unsqueeze(1) * degrees.unsqueeze(0) / (2 * m)

    # Calculate the modularity matrix B
    B = adj - expected_edges

    return B


def edge_index_to_adjacency(edge_index, num_nodes):
    # 创建一个全零的邻接矩阵
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # 根据边索引列表更新邻接矩阵
    for edge in edge_index.T:
        adjacency_matrix[edge[0], edge[1]] = 1
        adjacency_matrix[edge[1], edge[0]] = 1  # 如果是有向图，可以去掉这行

    return adjacency_matrix


def load_mat(data_path, dataset):
    data = sio.loadmat(data_path)
    if dataset == 'Amazon':
        label = data['gnd']
        feat = data['X']
        adj = data['A']
    else:
        label = data['Label']
        feat = np.array(data['Attributes'].todense())
        adj = data['Network']

    # 归一化处理
    adj_norm = normalize_adj(adj)
    data = {'adj_norm': adj_norm, 'adj': adj, 'feat': feat, 'label': label}

    return data


def get_adj(sparse_adjacency_matrix):
    max_index = max(max(index_pair) for index_pair in sparse_adjacency_matrix) + 1
    dimension = max_index  # 假设索引是从0开始的

    # 创建一个全零的二维矩阵
    adjacency_matrix = np.zeros((dimension, dimension), dtype=int)

    # 遍历稀疏矩阵中的每个元素
    for index_pair in sparse_adjacency_matrix:
        if index_pair.size == 2:  # 确保每个元素确实包含两个索引
            i, j = index_pair
            adjacency_matrix[i][j] = 1
    return adjacency_matrix


def load_npz(dataset):
    data = np.load("./dataset/{}.npz".format(dataset))
    feat = data['node_features']
    adj = get_adj(data['edges'])
    labels = data['node_labels']
    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_norm = adj_norm.toarray()
    return adj_norm, feat, labels, adj


def load_anomaly_detection_dataset(dataset, datadir='./data'):


    data = sio.loadmat(f'{datadir}/{dataset}.mat')
    truth = data['Label'] if ('Label' in data) else data['gnd']
    feat = data['Attributes'] if ('Attributes' in data) else data['X']
    adj = data['Network'] if ('Network' in data) else data['A']

    truth = truth.flatten()

    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_norm = adj_norm
    # adj = adj + sp.eye(adj.shape[0])
    if issparse(feat):

        feat = feat.toarray()

    return adj_norm, feat, truth, adj


def load_dataset(datadir, dataset):
    if dataset in ['weibo', 'reddit', 'reddit', 'disney', 'books', 'inj_cora']:
        #data = load_data(dataset)
        file_path = os.path.join(datadir, dataset+'.pt')
        data=torch.load(file_path)
        attrs = data.x.numpy().astype(np.float32)
        init_adj = data.edge_index
        # adj = to_scipy_sparse_matrix(init_adj).toarray()
        # adj = adj + adj.T
        adj_origin = edge_index_to_adjacency(init_adj, len(attrs))
        np.fill_diagonal(adj_origin, 1)
        adj_label = adj_origin
        adj = csr_matrix(adj_origin)

        label = data.y.bool()
        label = torch.where(label, 1.0, 0.0).numpy()

    elif dataset in ['tolokers', 'quest']:
        adj, attrs, label, adj_label = load_npz(dataset)
    else:
        adj_norm, attrs, label, adj = load_anomaly_detection_dataset(dataset)

    data = {'adj': adj, 'feat': attrs, 'label': label}
    return data


def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format."""
    nx_graph = nx.from_scipy_sparse_matrix(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph


def generate_rwr_subgraph(dgl_graph, subgraph_size):
    """Generate subgraph with RWR algorithm."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    traces, _ = dgl.sampling.random_walk(g=dgl_graph, nodes=all_idx, restart_prob=0.0,
                                         length=subgraph_size * 3)
    traces = traces[:, 1:]
    subv = []
    for i, trace in enumerate(traces):
        subv.append(torch.unique(trace, sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace, _ = dgl.sampling.random_walk(g=dgl_graph, nodes=[i], restart_prob=0.9,
                                                    length=subgraph_size * 5)
            subv[i] = torch.unique(cur_trace, sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= 2) and (retry_time > 10):
                subv[i] = (subv[i] * reduced_size)
        subv[i] = subv[i][:reduced_size]
        subv[i].append(i)

    return subv


def DS_Combin(alpha, classes):
    """
    :param alpha: All Dirichlet distribution parameters.
    :return: Combined Dirichlet distribution parameters.
    """

    def DS_Combin_two(classes, alpha1, alpha2):
        """
        :param alpha1: Dirichlet distribution parameters of view 1
        :param alpha2: Dirichlet distribution parameters of view 2
        :return: Combined Dirichlet distribution parameters
        """
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = classes / S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, classes, 1), b[1].view(-1, 1, classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate C
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        C = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))

        # calculate new S
        S_a = classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a

    for v in range(len(alpha) - 1):
        if v == 0:
            alpha_a = DS_Combin_two(classes, alpha[0], alpha[1])
        else:
            alpha_a = DS_Combin_two(classes, alpha_a, alpha[v + 1])
    return alpha_a


def macro_precision_recall_f1(y_true, y_scores):
    """
    Calculate Macro-Precision, Macro-Recall, and Macro-F1-Score.

    Parameters:
    - y_true: Ground truth labels (list or array-like)
    - y_pred: Predicted labels (list or array-like)

    Returns:
    - macro_precision: Macro precision score
    - macro_recall: Macro recall score
    - macro_f1_score: Macro F1 score
    """
    # Calculate precision for each class
    k = sum(1 for label in y_true if label == 1)

    # Create predicted labels based on the top k scores
    y_pred = [1 if score >= sorted(y_scores, reverse=True)[k - 1] else 0 for score in y_scores]

    # Calculate precision for each class
    precision_anomalous = precision_score(y_true, y_pred, labels=[1], average=None)
    precision_normal = precision_score(y_true, y_pred, labels=[0], average=None)

    # Calculate recall for each class
    recall_anomalous = recall_score(y_true, y_pred, labels=[1], average=None)
    recall_normal = recall_score(y_true, y_pred, labels=[0], average=None)

    # Calculate macro precision, recall, and f1 score
    macro_precision = (precision_anomalous + precision_normal) / 2
    macro_recall = (recall_anomalous + recall_normal) / 2
    macro_f1_score = f1_score(y_true, y_pred, average='macro')

    return macro_precision[0], macro_recall[0], macro_f1_score


if __name__ == "__main__":
    data = load_data('D:\Program\Pythonprogram\Trusted-GAD\Trusted-GAD\data', 'Enron')
    print('------------Test-----------')
