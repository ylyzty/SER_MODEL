import argparse
import time

import torch
import torch_geometric
import numpy as np

from torch_geometric.loader import DataLoader
from dag_transformer.models import GraphTransformer
from datasets.circuit_dataset import CircuitDataset
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

# args 设置
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='batch size')  # batch_size should >= 50
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0.2, help='dropout ratio')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=10, help='maximum number of epochs')
parser.add_argument('--conv_name', type=str, default='NO', help='conv')
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--DAG_attention', type=int, default=1)
parser.add_argument('--num_heads', type=int, default=4, help="number of heads")
parser.add_argument('--pe', type=str, default='dagpe')
parser.add_argument('--gps', type=int, default=0)
parser.add_argument('--SAT', type=bool, default=False)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--VN', type=int, default=0)
args = parser.parse_args()

path = "../circuit_data"
dataset = CircuitDataset(root=path)

args.num_classes = 1
args.num_features = 2
print(args)


def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def train(train_loader, test_loader, deg):
    model = GraphTransformer(in_size=args.num_features,
                             num_class=args.num_classes,
                             d_model=args.nhid,
                             gps=args.gps,
                             abs_pe=args.pe,
                             dim_feedforward=4 * args.nhid,
                             dropout=args.dropout_ratio,
                             num_heads=args.num_heads,
                             num_layers=args.num_layers,
                             batch_norm=True,
                             in_embed=False,
                             edge_embed=False,
                             SAT=args.SAT,
                             deg=deg
                             ).to(args.device)
    print(f"Total number of parameters: {count_parameters(model=model)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = torch.nn.MSELoss(reduction='mean')

    mse = 1e4
    mae = 1e4
    t = time.time()
    model.train()
    for epoch in range(args.epochs):
        loss_train = 0.0
        print(f"Epoch {epoch + 1}/{args.epochs}, LR {optimizer.param_groups[0]['lr']}")

        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            out = model(data)
            loss = criterion(out, data.y)

            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        train_loss = loss_train / len(train_loader)
        test_loss, preds, labels = test(test_loader, model, epoch, criterion)

        mse_tmp = mean_squared_error(labels, preds)
        mae_tmp = mean_absolute_error(labels, preds)
        if mse_tmp < mse:
            mse = mse_tmp
        if mae_tmp < mae:
            mae = mae_tmp
        print(f"Epoch: {epoch + 1}, train_loss: {train_loss}, "
              f"test_loss: {test_loss}, test_mse: {mse_tmp}, test_mae: {mae_tmp}")

        lr_scheduler.step()

    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))
    return mse, mae


def test(loader, model, epoch, criterion):
    model.eval()
    loss_test = 0.0
    Preds = []
    Labels = []

    for data in loader:
        data = data.to(args.device)
        out = model(data)
        loss = criterion(out, data.y)
        loss_test += loss.item()

        # preds = out.cpu().detach().numpy()
        preds = torch.squeeze(out).cpu().float().detach().numpy()
        # preds = preds.float().numpy()
        labels = torch.squeeze(data.y).cpu().float().detach().numpy()
        # labels = labels.float().numpy()
        Preds.append(preds)
        Labels.append(labels)

    return loss_test / len(loader), Preds, Labels


if __name__ == "__main__":
    dataset = CircuitDataset(root='../circuit_data', DAG=True)

    # 划分 训练集 和 测试集
    train_size = int(len(dataset) * 0.7)
    dataset_idx = np.arange(start=0, stop=len(dataset), step=1, dtype=int)
    dataset_idx = np.random.permutation(dataset_idx)
    train_idx = torch.tensor(dataset_idx[:train_size], dtype=torch.long)
    test_idx = torch.tensor(dataset_idx[train_size:], dtype=torch.long)
    train_loader = DataLoader(dataset[train_idx], batch_size=args.batch_size, shuffle=True)
    print(len(train_loader))
    test_loader = DataLoader(dataset[test_idx], batch_size=args.batch_size, shuffle=True)

    MSE = []
    MAE = []
    for r in range(args.runs):
        torch.manual_seed(r)
        np.random.seed(r)

        deg = torch.cat([
            torch_geometric.utils.degree(data.edge_index[1], num_nodes=data.num_nodes) for data in dataset])
        mse, mae = train(train_loader=train_loader, test_loader=test_loader, deg=deg)

        MSE.append(mse)
        MAE.append(mae)

    print(f"10 times MSE, MAE: {np.mean(MSE)}, {np.mean(MAE)}")
