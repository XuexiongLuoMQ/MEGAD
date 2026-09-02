import argparse
import random
import time

import networkx as nx
import numpy as np
import torch
import sklearn.metrics as skm
from networkx import modularity_matrix
import warnings
import os
import csv
from datetime import datetime

from model import TrustedGAD, reconstruction_loss
from utils import load_dataset, macro_precision_recall_f1

warnings.filterwarnings("ignore")


def arg_parse():
    parser = argparse.ArgumentParser(description='TGAD Full-Graph Arguments.')
    parser.add_argument('-dataset', default='weibo')
    parser.add_argument('--datadir', dest='datadir', default=r'./data/')
    parser.add_argument('--epochs', dest='epochs', default=100, type=int)
    parser.add_argument('--hidden-dim', dest='hidden_dim', default=128, type=int)
    parser.add_argument('-num_layers', type=int, default=2)
    parser.add_argument('--lambda-epochs', type=int, default=50, metavar='N')
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('--test-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)  # 保留，作为默认起始种子
    parser.add_argument('--num-runs', type=int, default=5)  # 新增：多随机种子轮数
    parser.add_argument('--weight_uncer', type=float, default=1)
    parser.add_argument('--weight_evide', type=float, default=1)
    parser.add_argument('--save-dir', type=str, default='./results')
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_train_test_split(num_nodes, test_ratio=0.2, seed=0):
    rng = np.random.RandomState(seed)
    all_idx = np.arange(num_nodes)
    rng.shuffle(all_idx)
    split = int((1 - test_ratio) * num_nodes)
    train_idx = all_idx[:split]
    test_idx = all_idx[split:]
    return train_idx, test_idx


def get_run_seeds(base_seed, num_runs):
    """
    基于起始 seed 生成多轮实验随机种子
    例如 base_seed=0, num_runs=5 -> [0, 1, 2, 3, 4]
    """
    return [base_seed + i for i in range(num_runs)]


def save_config_once(args, exp_name):
    """
    参数只保存一遍
    """
    os.makedirs(args.save_dir, exist_ok=True)
    config_path = os.path.join(args.save_dir, f"{exp_name}_config.csv")

    header = [
        'time',
        'exp_name',
        'dataset',
        'epochs',
        'hidden_dim',
        'num_layers',
        'lambda_epochs',
        'lr',
        'test_ratio',
        'num_runs',
        'seed_list'
    ]

    seed_list = get_run_seeds(args.seed, args.num_runs)
    row = [
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        exp_name,
        args.dataset,
        args.epochs,
        args.hidden_dim,
        args.num_layers,
        args.lambda_epochs,
        args.lr,
        args.test_ratio,
        args.num_runs,
        str(seed_list)
    ]

    with open(config_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)

    print(f"[SAVE] config has been saved to: {config_path}")


def save_run_results(exp_name, args, run_results):
    """
    保存每一轮随机种子的结果
    参数不重复保存，仅保留 run_id、seed 和指标
    """
    os.makedirs(args.save_dir, exist_ok=True)
    run_path = os.path.join(args.save_dir, f"{exp_name}_per_run.csv")

    header = [
        'run_id',
        'seed',
        'avg_epoch_time',
        'auc',
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'score_uncertainty_corr'
    ]

    with open(run_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for item in run_results:
            writer.writerow([
                item['run_id'],
                item['seed'],
                round(item['avg_epoch_time'], 6),
                round(item['auc'], 6),
                round(item['accuracy'], 6),
                round(item['precision'], 6),
                round(item['recall'], 6),
                round(item['f1_score'], 6),
                round(item['corr'], 6)
            ])

    print(f"[SAVE] per-run results have been saved to: {run_path}")


def save_summary_results(exp_name, args, run_results):
    """
    保存最终汇总结果：重点统计 AUC 均值和标准差
    同时汇总其他指标，便于实验表格整理
    """
    os.makedirs(args.save_dir, exist_ok=True)
    summary_path = os.path.join(args.save_dir, f"{exp_name}_summary.csv")

    auc_list = np.array([x['auc'] for x in run_results], dtype=np.float64)
    avg_time_list = np.array([x['avg_epoch_time'] for x in run_results], dtype=np.float64)
    acc_list = np.array([x['accuracy'] for x in run_results], dtype=np.float64)
    pre_list = np.array([x['precision'] for x in run_results], dtype=np.float64)
    rec_list = np.array([x['recall'] for x in run_results], dtype=np.float64)
    f1_list = np.array([x['f1_score'] for x in run_results], dtype=np.float64)
    corr_list = np.array([x['corr'] for x in run_results], dtype=np.float64)

    header = [
        'time',
        'exp_name',
        'dataset',
        'num_runs',
        'auc_mean',
        'auc_std',
        'avg_epoch_time_mean',
        'avg_epoch_time_std',
        'accuracy_mean',
        'accuracy_std',
        'precision_mean',
        'precision_std',
        'recall_mean',
        'recall_std',
        'f1_score_mean',
        'f1_score_std',
        'corr_mean',
        'corr_std'
    ]

    row = [
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        exp_name,
        args.dataset,
        len(run_results),
        round(np.mean(auc_list), 6),
        round(np.std(auc_list), 6),
        round(np.mean(avg_time_list), 6),
        round(np.std(avg_time_list), 6),
        round(np.mean(acc_list), 6),
        round(np.std(acc_list), 6),
        round(np.mean(pre_list), 6),
        round(np.std(pre_list), 6),
        round(np.mean(rec_list), 6),
        round(np.std(rec_list), 6),
        round(np.mean(f1_list), 6),
        round(np.std(f1_list), 6),
        round(np.mean(corr_list), 6),
        round(np.std(corr_list), 6)
    ]

    with open(summary_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)

    print(f"[SAVE] summary results have been saved to: {summary_path}")
    print(f"[FINAL SUMMARY] Dataset:{args.dataset} | AUC mean: {np.mean(auc_list):.4f} | AUC std: {np.std(auc_list):.4f}")


def run_single_seed(data, args, seed, run_id):
    print('\n' + '=' * 80)
    print(f'[RUN {run_id}/{args.num_runs}] seed = {seed}')
    print('=' * 80)

    setup_seed(seed)

    feature_dim = data['feat'].shape[1]
    node_num = data['feat'].shape[0]

    model = TrustedGAD(
        node_num=node_num,
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lambda_epochs=args.lambda_epochs
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if torch.cuda.is_available():
        model = model.cuda()

    print('-------Training model--------' + args.dataset + '-----------')

    X = data['feat'].astype(np.float32)
    adj = data['adj']
    label = data['label']
    print('-----------anomaly label number:' + str(np.sum(label)) + '-----------')

    adj_dense_np = adj.toarray().astype(np.float32)
    adj_B = modularity_matrix(nx.from_numpy_array(adj_dense_np)).astype(np.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X = torch.from_numpy(X).float().to(device)
    adj_tensor = torch.from_numpy(adj_dense_np).float().to(device)
    adj_B_tensor = torch.from_numpy(adj_B).float().to(device)
    label = torch.from_numpy(label).long().squeeze().to(device)

    num_nodes = X.shape[0]
    train_idx_np, test_idx_np = build_train_test_split(num_nodes, args.test_ratio, seed)
    train_idx = torch.from_numpy(train_idx_np).long().to(device)
    test_idx = torch.from_numpy(test_idx_np).long().to(device)

    train_start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        loss_evi, loss_rec, loss_attr_rec, loss_uncertainty, anom_score, u_final, alpha_s, alpha_a, alpha_attr, alpha_final = model(
            x=X,
            adj=adj_tensor,
            adj_B=adj_B_tensor,
            y=label,
            global_step=epoch + 1,
            train_idx=train_idx,
        )

        loss =args.weight_evide* torch.mean(loss_evi) + loss_rec + loss_attr_rec + args.weight_uncer*loss_uncertainty

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_score = anom_score[train_idx].squeeze()
            train_auc = skm.roc_auc_score(
                label[train_idx].detach().cpu().numpy(),
                train_score.detach().cpu().numpy()
            )

        print(
            f'Run:{run_id:02d} '
            f'Epoch:{epoch + 1:03d}/{args.epochs} '
            f'loss:{loss.item():.4f} '
            f'evi_loss:{torch.mean(loss_evi).item():.4f} '
            f'community_loss:{loss_rec.item():.4f} '
            f'train_auc:{train_auc:.4f}'
        )

    train_end_time = time.time()
    total_train_time = train_end_time - train_start_time
    avg_epoch_time = total_train_time / args.epochs

    print(f"[TIME] total_train_time: {total_train_time:.4f}s")
    print(f"[TIME] avg_epoch_time: {avg_epoch_time:.6f}s")

    print('-------Testing model--------' + args.dataset + '-----------')
    model.eval()

    with torch.no_grad():
        _, _, _, _, anom_score, u_final, _, _, _, _ = model(
            x=X,
            adj=adj_tensor,
            adj_B=adj_B_tensor,
            y=None,
            global_step=1,
            train_idx=None
        )

        y_true = label[test_idx].detach().cpu().numpy().squeeze()
        y_score = anom_score[test_idx].detach().cpu().numpy().squeeze()
        u_score = u_final[test_idx].detach().cpu().numpy().squeeze()

        auc = skm.roc_auc_score(y_true, y_score)
        precision, recall, f1_score = macro_precision_recall_f1(y_true, y_score)

        m = np.sum(y_true == 1)
        threshold_indices = np.argsort(-y_score)[:m]
        predicted_labels = np.zeros_like(y_true)
        predicted_labels[threshold_indices] = 1
        accuracy = np.mean(predicted_labels == y_true)

        print('[FINAL RESULT] Run:{} Seed:{} Dataset:{} AUC:{:.4f} ACC:{:.4f} Pre:{:.4f} Rec:{:.4f} F1:{:.4f}'.format(
            run_id, seed, args.dataset, auc, accuracy, precision, recall, f1_score
        ))

        corr = np.corrcoef(y_score.reshape(-1), u_score.reshape(-1))[0, 1]
        print(f'[ANALYSIS] score-uncertainty corr: {corr:.4f}')

    result = {
        'run_id': run_id,
        'seed': seed,
        'avg_epoch_time': avg_epoch_time,
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'corr': corr
    }

    return result


if __name__ == "__main__":
    args = arg_parse()

    os.makedirs(args.save_dir, exist_ok=True)

    # 数据只加载一次
    data = load_dataset(args.datadir, args.dataset)

    exp_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{args.dataset}_runs{args.num_runs}_{exp_time}"

    save_config_once(args, exp_name)

    # 多随机种子实验
    seed_list = get_run_seeds(args.seed, args.num_runs)
    run_results = []

    for i, seed in enumerate(seed_list, start=1):
        result = run_single_seed(data, args, seed, i)
        run_results.append(result)

    # 保存每轮结果
    save_run_results(exp_name, args, run_results)

    # 保存最终均值和标准差
    save_summary_results(exp_name, args, run_results)