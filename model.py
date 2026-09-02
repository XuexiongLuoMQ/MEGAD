import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import DS_Combin


def KL(alpha, c):
    beta = torch.ones((1, c), device=alpha.device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1

    label = F.one_hot(p.long(), num_classes=c).float()
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1.0, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = 0.01 * KL(alp, c)

    return A - B


def reconstruction_loss(B, B_hat):
    return F.binary_cross_entropy_with_logits(B_hat, B)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, adj):
        device = x.device
        n = adj.size(0)

        I = torch.eye(n, device=device, dtype=adj.dtype)
        A_tilde = adj + I
        deg = A_tilde.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)

        A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
        out = A_norm @ self.linear(x)
        return out


class AttributeEncoder(nn.Module):
    """
    参考 Dominant 的属性编码器：
    2层标准GCN编码
    X -> GCN -> GCN -> Z
    """

    def __init__(self, in_dim, hidden_dim, dropout=0.5):
        super(AttributeEncoder, self).__init__()
        self.gc1 = GCNLayer(in_dim, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, hidden_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        return x


class AttributeDecoder(nn.Module):
    """
    参考 Dominant 的属性解码器：
    2层标准GCN解码
    Z -> GCN -> GCN -> X_hat
    """

    def __init__(self, in_dim, hidden_dim, dropout=0.5):
        super(AttributeDecoder, self).__init__()
        self.gc1 = GCNLayer(hidden_dim, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, in_dim)
        self.dropout = dropout

    def forward(self, z, adj):
        z = F.relu(self.gc1(z, adj))
        z = F.dropout(z, self.dropout, training=self.training)
        x_hat = self.gc2(z, adj)
        return x_hat


class AttributeReconGAE(nn.Module):
    """
    属性重构模块：
    X -> 2层标准GCN编码 -> Z -> 2层标准GCN解码 -> X_hat
    """

    def __init__(self, in_dim, hidden_dim, dropout=0.5):
        super(AttributeReconGAE, self).__init__()
        self.encoder = AttributeEncoder(in_dim, hidden_dim, dropout)
        self.decoder = AttributeDecoder(in_dim, hidden_dim, dropout)

    def forward(self, x, adj):
        z = self.encoder(x, adj)  # [N, H]
        x_hat = self.decoder(z, adj)  # [N, F]
        return x_hat, z


class CommunityAutoencoder(nn.Module):
    def __init__(self, input_dim, n_enc_1, n_enc_2, n_enc_3, output_dim):
        super(CommunityAutoencoder, self).__init__()
        self.encoder = nn.ModuleList([
            nn.Linear(input_dim, n_enc_1, dtype=torch.float32),
            nn.Dropout(p=0.3),
            nn.Linear(n_enc_1, n_enc_3, dtype=torch.float32)
        ])

        self.decoder = nn.ModuleList([
            nn.Linear(n_enc_3, n_enc_1, dtype=torch.float32),
            nn.Dropout(p=0.3),
            nn.Linear(n_enc_1, output_dim, dtype=torch.float32)
        ])
        self.W11 = nn.Parameter(torch.Tensor(n_enc_3, n_enc_2))
        nn.init.xavier_uniform_(self.W11)

    def forward(self, x):
        encoded = x
        for layer in self.encoder:
            encoded = F.relu(layer(encoded))

        decoded = encoded
        for layer in self.decoder:
            decoded = F.relu(layer(decoded))

        encoder = torch.matmul(encoded, self.W11)
        return decoded, encoder


class AnomalyAwareGCN(nn.Module):
    """
    保持原始思想：
        h_i = sigma(W1 h_i + sum_{j in N_i} phi_{i,j} W2(h_i - h_j))

    复杂度优化：
    1. 不再构造全节点对差分张量 [N, N, F]
    2. 只在真实边上计算差分，复杂度从 O(N^2 d) 降为 O(E d)
    3. 利用 W2(h_i-h_j)=W2h_i-W2h_j，先线性变换，再做边差分
    """

    def __init__(self, input_dim, hidden_dim, add_self_loops=True):
        super(AnomalyAwareGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.add_self_loops = add_self_loops

        self.W1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.a = nn.Linear(hidden_dim, 1, bias=False)

        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.xavier_uniform_(self.a.weight)

    @staticmethod
    def edge_softmax(dst, logits, num_nodes):
        """
        对每个目标节点 dst 的入边进行 softmax 归一化
        dst:    [E]
        logits: [E]
        """
        max_per_dst = torch.full(
            (num_nodes,),
            -1e15,
            device=logits.device,
            dtype=logits.dtype
        )
        max_per_dst.scatter_reduce_(0, dst, logits, reduce="amax", include_self=True)

        exp_logits = torch.exp(logits - max_per_dst[dst])

        denom = torch.zeros(num_nodes, device=logits.device, dtype=logits.dtype)
        denom.scatter_add_(0, dst, exp_logits)

        alpha = exp_logits / (denom[dst] + 1e-12)
        return alpha

    def get_edge_index(self, A):
        """
        A 支持 dense tensor 或 sparse tensor
        返回:
            src, dst, edge_weight
        表示 src -> dst 的边
        """
        if A.is_sparse:
            A = A.coalesce()
            idx = A.indices()
            val = A.values()
            src, dst = idx[0], idx[1]
            edge_weight = val.float()
        else:
            idx = torch.nonzero(A > 0, as_tuple=False)
            src, dst = idx[:, 0], idx[:, 1]
            edge_weight = A[src, dst].float()

        return src.long(), dst.long(), edge_weight

    def forward(self, X, A, k=2):
        """
        为保持与原 GCN 接口一致，保留参数 k，但此处不显式使用。
        输入:
            X: [N, F]
            A: [N, N]
        输出:
            H: [N, H]
        """
        assert X.dim() == 2, "AnomalyAwareGCN 在整图训练下要求 X 为 [N, F]"
        num_nodes = X.size(0)

        src, dst, edge_weight = self.get_edge_index(A)

        if self.add_self_loops:
            loop = torch.arange(num_nodes, device=X.device)
            src = torch.cat([src, loop], dim=0)
            dst = torch.cat([dst, loop], dim=0)
            loop_weight = torch.ones(num_nodes, device=X.device, dtype=X.dtype)
            edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        # 自身线性项 W1 h_i
        h_self = self.W1(X)  # [N, H]

        # 先做 W2，再在边上做差分
        z = self.W2(X)  # [N, H]

        # 按公式 Delta_{i,j} = h_i - h_j
        # 若 dst=i, src=j，则应写作 z[dst] - z[src]
        delta_z = z[dst] - z[src]  # [E, H]

        # a^T W2 Delta_{i,j}
        logits = self.a(delta_z).squeeze(-1)  # [E]
        logits = torch.sigmoid(logits)

        # 若邻接为带权图，则将边权纳入
        logits = logits + torch.log(edge_weight + 1e-12)

        # 对每个节点 i 的邻居 j 做归一化
        alpha = self.edge_softmax(dst, logits, num_nodes)  # [E]

        # 聚合差分消息
        msg = alpha.unsqueeze(-1) * delta_z  # [E, H]

        h_neigh = torch.zeros(
            num_nodes, self.hidden_dim,
            device=X.device, dtype=X.dtype
        )
        h_neigh.index_add_(0, dst, msg)

        h = F.elu(h_self + h_neigh)
        return h


class TrustedGAD(nn.Module):
    def __init__(self, node_num, feature_dim, hidden_dim, num_layers, lambda_epochs=1):
        super(TrustedGAD, self).__init__()
        self.lambda_epochs = lambda_epochs
        self.classes = 2

        self.community_ae = CommunityAutoencoder(
            input_dim=node_num,
            n_enc_1=256,
            n_enc_2=feature_dim,
            n_enc_3=128,
            output_dim=node_num
        )
        self.attr_gae = AttributeReconGAE(feature_dim, hidden_dim, dropout=0.2)

        # 这里仅替换为优化后的 GCN，外部接口保持不变
        self.gcn = nn.ModuleList([
            AnomalyAwareGCN(feature_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        self.evidence_head_s = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Dropout(p=0.3),
            nn.Softplus()
        )

        self.evidence_head_a = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Dropout(p=0.3),
            nn.Softplus()
        )

        self.evidence_head_attr = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Dropout(p=0.3),
            nn.Softplus()
        )

        # 三个分支，每个分支输入是 [alpha(2维) + u(1维)] = 3维
        # 三个分支拼接后总维度 = 3 * 3 = 9
        self.expert_gate = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Dropout(p=0.3),
            nn.Softplus()
        )

    def computebeliefs_and_uncertainty(self, alpha):
        total_evidence = torch.sum(alpha, dim=1, keepdim=True)
        belief_mass = (alpha - 1) / (total_evidence + 1e-12)
        uncertainty_mass = self.classes / (total_evidence + 1e-12)
        return belief_mass, uncertainty_mass

    def encode_two_views(self, x, x_c, adj):
        h_s = x_c
        h_a = x
        for layer in self.gcn:
            h_s = layer(h_s, adj)
            h_a = layer(h_a, adj)
        return h_s, h_a

    def forward(self, x, adj, adj_B, y, global_step=1, train_idx=None):

        adj_B_hat, x_c = self.community_ae(adj_B)
        loss_rec = reconstruction_loss(adj_B, adj_B_hat)

        # 属性分支
        x_hat, embeddings_attr = self.attr_gae(x, adj)
        attr_error = torch.mean((x_hat - x) ** 2, dim=1, keepdim=True)
        #print(attr_error.shape,'11111111111')
        loss_attr_rec = torch.mean(torch.sum((x_hat - x) ** 2, dim=1))

        embeddings_s, embeddings_a = self.encode_two_views(x, x_c, adj)

        evidence_s = self.evidence_head_s(embeddings_s)
        evidence_a = self.evidence_head_a(embeddings_a)
        evidence_attr = self.evidence_head_attr(attr_error)

        alpha_s = evidence_s + 1
        alpha_a = evidence_a + 1
        alpha_attr = evidence_attr + 1

        _, u_s = self.computebeliefs_and_uncertainty(alpha_s)
        _, u_a = self.computebeliefs_and_uncertainty(alpha_a)

        # 属性分支
        _, u_attr = self.computebeliefs_and_uncertainty(alpha_attr)

        # 每个分支内部拼接：证据 + 不确定性
        branch_s = torch.cat([evidence_s, u_s], dim=1)  # [N, 3]
        branch_a = torch.cat([evidence_a, u_a], dim=1)  # [N, 3]
        branch_attr = torch.cat([evidence_attr, u_attr], dim=1)  # [N, 3]

        # 三个专家拼接
        gate_input = torch.cat([branch_s, branch_a, branch_attr], dim=1)  # [N, 9]

        # 两层 MLP 输出专家权重
        gate_logits = self.expert_gate(gate_input)  # [N, 3]
        expert_weights = torch.softmax(gate_logits, dim=1)  # [N, 3]

        w_s = expert_weights[:, 0:1]  # [N, 1]
        w_a = expert_weights[:, 1:2]  # [N, 1]
        w_attr = expert_weights[:, 2:3]  # [N, 1]

        # 不确定性分类损失
        expert_uncertainties = torch.cat([u_s, u_a, u_attr], dim=1)  # [N, 3]
        loss_uncertainty = torch.mean(torch.sum(expert_weights * expert_uncertainties, dim=1))
        
         # 专家平衡损失
        #avg_weights = expert_weights.mean(dim=0)                 # [3]
        #target = torch.tensor([1/3, 1/3, 1/3], device=expert_weights.device)
        #loss_balance = ((avg_weights - target) ** 2).sum()

        evidence_final = w_s * evidence_s + w_a * evidence_a + w_attr * evidence_attr  # [N, 2]
        # _, u_final = self.computebeliefs_and_uncertainty(alpha_final)
        alpha_final = evidence_final + 1

        # 用DS融合，得到最终的不确定性
        alpha_DS_for_u = DS_Combin([alpha_s, alpha_a, alpha_attr], self.classes)
        _, u_final = self.computebeliefs_and_uncertainty(alpha_DS_for_u)
        p_anom = alpha_final[:, 1:2] / (alpha_final.sum(dim=1, keepdim=True) + 1e-12)
        #print(p_anom, u_final,'2222222222')
        #anom_score = p_anom * (1 - u_final)
        anom_score = p_anom * u_final


        loss_evi = None
        if y is not None and train_idx is not None:
            loss_s = ce_loss(y[train_idx], alpha_s[train_idx], self.classes, global_step, self.lambda_epochs)
            loss_a = ce_loss(y[train_idx], alpha_a[train_idx], self.classes, global_step, self.lambda_epochs)
            loss_attr = ce_loss(y[train_idx], alpha_attr[train_idx], self.classes, global_step, self.lambda_epochs)
            loss_final = ce_loss(y[train_idx], alpha_final[train_idx], self.classes, global_step, self.lambda_epochs)

            loss_evi = loss_s + loss_a + loss_attr + loss_final

        return loss_evi, loss_rec, loss_attr_rec, loss_uncertainty, anom_score, u_final, alpha_s, alpha_a, alpha_attr, alpha_final
