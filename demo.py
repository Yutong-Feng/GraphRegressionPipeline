import os

import numpy as np
import pandas as pd
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch_geometric import nn as pygnn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from sklearn.metrics import f1_score, top_k_accuracy_score


class Config:
    def __init__(self) -> None:
        self.device = "cuda" if th.cuda.is_available() else "cpu"
        self.epoches = 10
        self.batch_size = 128
        self.lr = 1e-3
        self.beta = 2

        self.emb_dim = 8
        self.K = 2
        self.hid_dim0 = 64
        self.graph_feats = 64
        self.hid_dim1 = 32


class GraphDataset(Dataset):
    def __init__(self, data_name, data_type) -> None:
        super(GraphDataset, self).__init__()
        file_head = "npz_all\\npz"
        self.data_paths = []

        assert data_name in ["tile", "layout"]
        assert data_type in ["train", "valid", "test"]

        task_types = {"tile": ["xla"], "layout": ["nlp", "xla"]}[data_name]
        sample_types = {
            "tile": [""],
            "layout": ["default", "random"]
        }[data_name]

        for task_type in task_types:
            for sample_type in sample_types:
                path = os.path.join(file_head, data_name, task_type,
                                    sample_type, data_type)
                file_list = os.listdir(path)
                print(f"Found {len(file_list)} files in {path}.")
                self.data_paths += list(
                    map(lambda x: os.path.join(path, x), file_list))
        print(f"Total {len(self.data_paths)} files. Data is Ready.")

    def __getitem__(self, index):
        raw_data = dict(np.load(self.data_paths[index]))

        node_feat = th.from_numpy(raw_data["node_feat"].astype(np.float32))
        edge_index = th.from_numpy(raw_data["edge_index"].astype(np.int64).T)
        node_op = th.from_numpy(raw_data["node_opcode"].astype(np.int32))
        config_feats = th.from_numpy(raw_data["config_feat"].astype(
            np.float32))

        y = (raw_data['config_runtime'] /
             raw_data['config_runtime_normalizers']).astype(np.float32)
        y = (y - y.min()) / (y.max() - y.min())
        y = th.from_numpy(y)
        y = (y == y.min()).float()

        return Data(x=node_feat,
                    edge_index=edge_index,
                    y=y,
                    node_op=node_op,
                    config_feats=config_feats)
        # return node_feat,edge_index,y

    def __len__(self):
        return len(self.data_paths)


class ChebNet(th.nn.Module):
    def __init__(self, emb_dim, K, hid_dim0, graph_feats, hid_dim1):
        super().__init__()
        self.embedding = nn.Embedding(120, emb_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        in_dim = emb_dim + 140
        self.conv1 = pygnn.ChebConv(in_dim, hid_dim0, K)
        self.norm1 = nn.BatchNorm1d(hid_dim0)
        self.conv2 = pygnn.ChebConv(hid_dim0, graph_feats, K)
        self.norm2 = nn.BatchNorm1d(graph_feats)
        self.activ = nn.Tanh()

        self.dense = nn.Sequential(nn.Linear(graph_feats + 24,
                                             hid_dim1), self.activ,
                                   nn.Linear(hid_dim1, hid_dim1), self.activ,
                                   nn.Linear(hid_dim1, 1), nn.Sigmoid())

    def forward(self, x_cfg: th.Tensor, x_feat: th.Tensor, x_op: th.Tensor,
                edge_index: th.Tensor) -> th.Tensor:

        #get graph features
        x_feat = (x_feat - x_feat.min(axis=1)[0].unsqueeze(1).repeat(
            1, 140)) / (x_feat.max(axis=1)[0].unsqueeze(1).repeat(1, 140) -
                        x_feat.min(axis=1)[0].unsqueeze(1).repeat(1, 140))
        x = th.concat([x_feat, self.embedding(x_op)], dim=1)

        #pass though conv layers
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = self.activ(x)

        # get 1d graph embedding using average pooling
        x_graph = th.mean(x, 0)

        #put graph data into config data
        x = th.concat([x_cfg, x_graph.repeat((len(x_cfg), 1))], axis=1)
        #put into dense nn
        x = self.dense(x)
        return x


class FScoreLoss(th.nn.Module):
    def __init__(self, beta=1, eps=1e-7):
        super().__init__()
        # beta=0.5 if precision is more important else beta=2
        self.beta = beta
        self.eps = eps

    def forward(
        self,
        y_true,
        y_pred,
    ):
        assert (y_true.shape == y_pred.shape)
        tp = (y_true * y_pred).sum().to(th.float32)
        fn = ((1 - y_true) * y_pred).sum().to(th.float32)
        fp = (y_true * (1 - y_pred)).sum().to(th.float32)
        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)

        f_score_loss = (1 + self.beta**2) * (precision * recall) / (
            (self.beta**2) * precision + recall + self.eps)

        # print(
        #     f'precision = {precision}, recall = {recall}, f_score = {f_score_loss}'
        # )

        return -1 * th.log(f_score_loss)


if __name__ == "__main__":
    config = Config()
    print(config.device)

    train_dataset = GraphDataset("tile", "train")
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  drop_last=True,
                                  shuffle=True)
    valid_dataset = GraphDataset("tile", "valid")
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=1,
                                  drop_last=True,
                                  shuffle=True)

    model = ChebNet(emb_dim=config.emb_dim,
                    K=config.K,
                    hid_dim0=config.hid_dim0,
                    graph_feats=config.graph_feats,
                    hid_dim1=config.hid_dim1).to(config.device)
    optimizer = th.optim.Adam(model.parameters(),
                              lr=config.lr,
                              weight_decay=0.01)
    criterion = FScoreLoss(beta=config.beta)
    running_loss = 0.0
    sample_num = 0

    model = model.train()
    for epoch in range(config.epoches):
        for graph in tqdm(train_dataloader):
            config_feats = graph.config_feats.to(config.device)

            node_feats = graph.x.to(config.device)
            node_op = graph.node_op.to(config.device)
            edge_index = graph.edge_index.to(config.device)
            target = graph.y.to(config.device)

            y_pred = model(config_feats, node_feats, node_op, edge_index)

            loss = criterion(target, y_pred.squeeze())
            # print(FScoreLoss().forward(target, y_pred.squeeze()))
            # nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            optimizer.step()
            loss.backward()

            sample_num += 1
            running_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {running_loss/sample_num}")
        model = model.eval()
        with th.no_grad():
            score = 0.0
            for graph in tqdm(valid_dataloader):
                config_feats = graph.config_feats.to(config.device)

                node_feats = graph.x.to(config.device)
                node_op = graph.node_op.to(config.device)
                edge_index = graph.edge_index.to(config.device)
                target = graph.y.to(config.device)

                y_pred = model(config_feats, node_feats, node_op, edge_index)
                # top_k_accuracy_score(target.cpu().numpy(),y_pred.cpu().squeeze().numpy(),k=2)
                print("")
