"""
Paper: Self-supervised Graph Learning for Recommendation
Author: Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, Jianxun Lian, and Xing Xie
Reference: https://github.com/wujcan/Rec-Torch
"""

__author__ = "Jiancan Wu"
__email__ = "wujcan@gmail.com"

__all__ = ["Rec"]

import torch,nni
from torch.serialization import save
import torch.sparse as torch_sp
import torch.nn as nn
import torch.nn.functional as F
from model.base import AbstractRecommender
from util.pytorch import inner_product, l2_loss
from util.pytorch import get_initializer
from util.common import Reduction
from data import PointwiseSamplerV2, PairwiseSamplerV2
import numpy as np
from time import time
from reckit import timer
import scipy.sparse as sp
from util.common import normalize_adj_matrix, ensureDir
from util.pytorch import sp_mat_to_sp_tensor
from reckit import randint_choice
import random
from torch.utils.data.sampler import  WeightedRandomSampler
from torch_geometric.utils import degree,to_undirected
class _LightGCN(nn.Module):
    def __init__(self, num_users, num_items, user_com,item_com,embed_dim, norm_adj, n_layers):
        super(_LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.user_com = user_com
        self.item_com = item_com
        self.embed_dim = embed_dim
        self.norm_adj = norm_adj
        self.n_layers = n_layers
        self.user_embeddings = nn.Embedding(self.num_users, self.embed_dim)
        self.item_embeddings = nn.Embedding(self.num_items, self.embed_dim)
        self.dropout = nn.Dropout(0.1)
        self._user_embeddings_final = None
        self._item_embeddings_final = None

        # # weight initialization
        # self.reset_parameters()

    def reset_parameters(self, pretrain=0, init_method="uniform", dir=None):
        if pretrain:
            pretrain_user_embedding = np.load(dir + 'user_embeddings.npy')
            pretrain_item_embedding = np.load(dir + 'item_embeddings.npy')
            pretrain_user_tensor = torch.FloatTensor(pretrain_user_embedding).cuda()
            pretrain_item_tensor = torch.FloatTensor(pretrain_item_embedding).cuda()
            self.user_embeddings = nn.Embedding.from_pretrained(pretrain_user_tensor)
            self.item_embeddings = nn.Embedding.from_pretrained(pretrain_item_tensor)
        else:
            init = get_initializer(init_method)
            init(self.user_embeddings.weight)
            init(self.item_embeddings.weight)

    def forward(self, sub_graph1, sub_graph2, uc,ic,com_graph,users, items, neg_items):
        user_embeddings, item_embeddings = self._forward_gcn(self.norm_adj)
        if sub_graph1 is None:
            user_embeddings1, item_embeddings1 =None,None
        else:
            user_embeddings1, item_embeddings1 = self._forward_gcn(sub_graph1)
        if sub_graph2 is None:
            user_embeddings2, item_embeddings2 = None, None
        else:
            user_embeddings2, item_embeddings2 = self._forward_gcn(sub_graph2)
        user_embeddings3, item_embeddings3 = self._forward_gcn2(uc,ic,com_graph)

        if user_embeddings1 is None:
            user_embeddingsf=user_embeddings
            item_embeddingsf=item_embeddings
        else:
            user_embeddingsf = torch.cat((user_embeddings, user_embeddings1), dim=1)
            item_embeddingsf =torch.cat((item_embeddings, item_embeddings1), dim=1)

        if user_embeddings2 is None:
            user_embeddingsf=user_embeddingsf
            item_embeddingsf=item_embeddingsf
        else:
            user_embeddingsf = torch.cat((user_embeddingsf, user_embeddings2), dim=1)
            item_embeddingsf =torch.cat((item_embeddingsf, item_embeddings2), dim=1)
        user_embeddingsf = torch.cat((user_embeddingsf, user_embeddings3), dim=1)
        item_embeddingsf = torch.cat((item_embeddingsf, item_embeddings3), dim=1)
        user_embs = F.embedding(users, user_embeddingsf)
        item_embs = F.embedding(items, item_embeddingsf)
        neg_item_embs = F.embedding(neg_items, item_embeddingsf)

        sup_pos_ratings = inner_product(user_embs, item_embs)  # [batch_size]
        sup_neg_ratings = inner_product(user_embs, neg_item_embs)  # [batch_size]
        sup_logits = sup_pos_ratings - sup_neg_ratings  # [batch_size]

        return sup_logits

    def _forward_gcn(self, norm_adj):
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            if isinstance(norm_adj, list):
                ego_embeddings = torch_sp.mm(norm_adj[k], ego_embeddings)
            else:
                ego_embeddings = torch_sp.mm(norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)

        return user_embeddings, item_embeddings

    def _forward_gcn2(self, uc,ic,norm_adj):
        user_com_feature = torch.matmul(uc.transpose(0, 1),
                                        self.user_embeddings.weight)
        item_com_feature = torch.matmul(ic.transpose(0, 1),
                                        self.item_embeddings.weight)
        ego_embeddings = torch.cat([user_com_feature, item_com_feature], dim=0)
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            if isinstance(norm_adj, list):
                ego_embeddings = torch_sp.mm(norm_adj[k], ego_embeddings)
            else:
                ego_embeddings = torch_sp.mm(norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_com_embeddings, item_com_embeddings = torch.split(all_embeddings, [self.user_com, self.item_com], dim=0)
        user_embeddings = torch.matmul(uc, user_com_embeddings)
        item_embeddings = torch.matmul(ic, item_com_embeddings)
        return user_embeddings, item_embeddings

    def predict(self, users):
        if self._user_embeddings_final is None or self._item_embeddings_final is None:
            raise ValueError("Please first switch to 'eval' mode.")
        user_embs = F.embedding(users, self._user_embeddings_final)
        temp_item_embs = self._item_embeddings_final
        ratings = torch.matmul(user_embs, temp_item_embs.T)
        return ratings

    def eval(self,sub_graph1, sub_graph2, uc,ic,com_graph):
        super(_LightGCN, self).eval()
        user_embeddings, item_embeddings = self._forward_gcn(self.norm_adj)
        if sub_graph1 is None:
            user_embeddings1, item_embeddings1 = None, None
        else:
            user_embeddings1, item_embeddings1 = self._forward_gcn(sub_graph1)
        if sub_graph2 is None:
            user_embeddings2, item_embeddings2 = None, None
        else:
            user_embeddings2, item_embeddings2 = self._forward_gcn(sub_graph2)
        user_embeddings3, item_embeddings3 = self._forward_gcn2(uc, ic, com_graph)
        if user_embeddings1 is None:
            user_embeddingsf = user_embeddings
            item_embeddingsf = item_embeddings
        else:
            user_embeddingsf = torch.cat((user_embeddings, user_embeddings1), dim=1)
            item_embeddingsf = torch.cat((item_embeddings, item_embeddings1), dim=1)

        if user_embeddings2 is None:
            user_embeddingsf = user_embeddingsf
            item_embeddingsf = item_embeddingsf
        else:
            user_embeddingsf = torch.cat((user_embeddingsf, user_embeddings2), dim=1)
            item_embeddingsf = torch.cat((item_embeddingsf, item_embeddings2), dim=1)
        self._user_embeddings_final= torch.cat((user_embeddingsf, user_embeddings3), dim=1)
        self._item_embeddings_final= torch.cat((item_embeddingsf, item_embeddings3), dim=1)

class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, u,i,adj):
        super(EstimateAdj, self).__init__()
        self.estimated_adj = nn.Parameter(torch.FloatTensor(u,i))
        self._init_estimation(adj)
    def _init_estimation(self, adj):
        with torch.no_grad():
            self.estimated_adj.data.copy_(adj)
    def forward(self):
        return self.estimated_adj

class Rec(AbstractRecommender):
    def __init__(self, config):
        super(Rec, self).__init__(config)

        self.config = config
        self.model_name = config["recommender"]
        self.dataset_name = config["dataset"]

        # General hyper-parameters
        self.reg = config['reg']
        self.reg1 = config['reg1']
        self.reg2 = config['reg2']
        self.reg3 = config['reg3']
        self.emb_size = config['embed_size']
        self.user_com = config['user_com']
        self.item_com = config['item_com']
        self.batch_size = config['batch_size']
        self.test_batch_size = config['test_batch_size']
        self.epochs = config["epochs"]
        self.verbose = config["verbose"]
        self.stop_cnt = config["stop_cnt"]
        self.learner = config["learner"]
        self.lr = config['lr']
        self.param_init = config["param_init"]

        # Hyper-parameters for GCN
        self.n_layers = config['n_layers']

        # Hyper-parameters for SSL
        self.aug_type = config["aug_type"]
        self.paug_type = config["paug_type"]
        self.probu = config["probu"]
        self.probi = config["probi"]
        self.proba = config["proba"]
        self.probd = config["probd"]
        self.thr = config["thr"]
        self.drop_percent = config["drop_percent"]
        self.p = config["p"]
        self.threshold = config["threshold"]
        self.thre= config["thre"]
        # Other hyper-parameters
        self.best_epoch = 0
        self.best_result = np.zeros([2], dtype=float)

        self.model_str = '#layers=%d-reg=%.0e' % (
            self.n_layers,
            self.reg
        )
        self.pretrain_flag = config["pretrain_flag"]
        if self.pretrain_flag:
            self.epochs = 0
        self.save_flag = config["save_flag"]
        self.save_dir, self.tmp_model_dir = None, None
        if self.pretrain_flag or self.save_flag:
            self.tmp_model_dir = config.data_dir + '%s/model_tmp/%s/%s/' % (
                self.dataset_name,
                self.model_name,
                self.model_str)
            self.save_dir = config.data_dir + '%s/pretrain-embeddings/%s/n_layers=%d/' % (
                self.dataset_name,
                self.model_name,
                self.n_layers,)
            ensureDir(self.tmp_model_dir)
            ensureDir(self.save_dir)

        self.load_dir = config.data_dir + '%s/pretrain-embeddings/DisCom/%s/n_layers=%d/' % (
                self.dataset_name,
                self.paug_type,
                self.n_layers,)

        self.num_users, self.num_items, self.num_ratings = self.dataset.num_users, self.dataset.num_items, self.dataset.num_train_ratings

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        adj_matrix = self.create_adj_mat(None,None,None)
        adj_matrix = sp_mat_to_sp_tensor(adj_matrix).to(self.device)

        self.lightgcn = _LightGCN(self.num_users, self.num_items, self.user_com,self.item_com,self.emb_size,
                                  adj_matrix, self.n_layers).to(self.device)

        if self.pretrain_flag:
            self.lightgcn.reset_parameters(pretrain=self.pretrain_flag, dir=self.save_dir)
        else:
            self.lightgcn.reset_parameters(init_method=self.param_init)
        self.optimizer = torch.optim.Adam(self.lightgcn.parameters(), lr=self.lr)
        if self.aug_type=='ProGNN':
            self.eadj =  EstimateAdj(self.num_users,self.num_items,self.adj).to(self.device)
            self.optimizer = torch.optim.Adam(list(self.lightgcn.parameters())+list(self.eadj.parameters()), lr=self.lr)


    def Bernoulli(self,rate):
        # num = random.randint(0, 1000000)
        num = randint_choice(1000000, size=1, replace=False)
        p = num / 1000000
        if p < rate:
            return 1
        else:
            return 0

    def drop_edge_weighted(self,edge_index, edge_weights, p: float, threshold: float = 1.):
        edge_weights = edge_weights / edge_weights.mean() * p
        edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
        sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

        return edge_index[:, sel_mask]

    def degree_drop_weights(self,users_items):
        adj=torch.from_numpy(users_items).t().long()
        all_edges = to_undirected(adj)
        deg = degree(all_edges[0], num_nodes=self.num_users + self.num_items)
        deg_u = deg[adj[0]].to(torch.float32)
        deg_i = deg[adj[1]].to(torch.float32)
        s_col = torch.log((deg_i + deg_u) / 2)
        weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

        return weights


    def calArray2dDiff(self,array_0, array_1):
        array_0_rows = array_0.view([('', array_0.dtype)] * array_0.shape[1])
        array_1_rows = array_1.view([('', array_1.dtype)] * array_1.shape[1])

        return np.setdiff1d(array_0_rows, array_1_rows).view(array_0.dtype).reshape(-1, array_0.shape[1])


    @timer
    def create_adj_mat(self, uc,ic,cui,is_subgraph=False, aug_type=None):
        n_nodes = self.num_users + self.num_items
        users_items = self.dataset.train_data.to_user_item_pairs()
        users_np, items_np = users_items[:, 0], users_items[:, 1]
        real_adj = {}
        if is_subgraph:
            ucu = np.argmax(uc, axis=1)
            icu = np.argmax(ic, axis=1)
            if aug_type == 'node-e':
                user_np = []
                item_np = []
                for edge in  users_items:
                    user = edge[0]
                    item = edge[1]
                    if self.Bernoulli(self.probu):
                        ucom = ucu[user]
                        up = uc[:, ucom]
                        up = np.delete(up, user, axis=0)
                        user = list(WeightedRandomSampler(up, 1, True))[0]

                    elif self.Bernoulli(self.probi):
                        icom = icu[item]
                        ip = ic[:, icom]
                        ip = np.delete(ip, item, axis=0)
                        item = list(WeightedRandomSampler(ip, 1, True))[0]
                    user_np.append(user)
                    item_np.append(item)
                user_np=np.array(user_np)
                item_np = np.array(item_np)
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_users)), shape=(n_nodes, n_nodes))

            if  aug_type == 'edge-e':
                user_np = []
                item_np = []
                lusers_items = users_items.tolist()
                for edge in lusers_items:
                    if edge[0] not in real_adj:
                        real_adj[edge[0]] = {}
                    real_adj[edge[0]][edge[1]] = 1
                for user in range(self.num_users):
                    for item in range(self.num_items):
                        ucom = ucu[user]
                        icom = icu[item]
                        uic = cui[ucom][icom]
                        if real_adj[user].get(item, "zxczxc") is "zxczxc":
                            # print([user, item])
                            if uic >= self.thr and self.Bernoulli(self.proba):
                                user_np.append(user)
                                item_np.append(item)
                        else:
                            if uic >=self.thr or not self.Bernoulli(self.probd):
                                user_np.append(user)
                                item_np.append(item)
                # percent = self.drop_percent
                # add_drop_num = int(len(users_np) * percent)
                # edge_idx = [i for i in range(len(users_np))]
                # drop_idx = random.sample(edge_idx, add_drop_num)
                # user_np = []
                # item_np = []
                # for i in range(0, len(users_np)):
                #     ucom = ucu[users_np[i]]
                #     icom = icu[items_np[i]]
                #     uic = cui[ucom][icom]
                #
                #     if i not in drop_idx:
                #         user_np.append(users_np[i])
                #         item_np.append(items_np[i])
                #     else:
                #         if uic >=self.thr or not self.Bernoulli(self.probd):
                #             user_np.append(users_np[i])
                #             item_np.append(items_np[i])
                #
                # l = [(i, j) for i in range(self.num_users) for j in range(self.num_items)]
                # l=self.calArray2dDiff(np.array(l),users_items)
                # add_list = random.sample(l, add_drop_num)
                # for i in add_list:
                #     ucom = ucu[i[0]]
                #     icom = icu[i[1]]
                #     uic = cui[ucom][icom]
                #     if uic >= self.thr and self.Bernoulli(self.proba):
                #         user_np.append(i[0])
                #         item_np.append(i[1])

                user_np = np.array(user_np)
                item_np = np.array(item_np)
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_users)), shape=(n_nodes, n_nodes))

            if  aug_type == 'com':
                user_np = []
                item_np = []
                for user in range(self.user_com):
                    for item in range(self.item_com):
                        uic = cui[user][item]
                        if uic >= self.thr:
                            user_np.append(user)
                            item_np.append(item)
                user_np = np.array(user_np)
                item_np = np.array(item_np)
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_com)), shape=(self.user_com+self.item_com, self.user_com+self.item_com))

            if aug_type == 'GraphCL':
                percent = self.drop_percent / 2
                add_drop_num = int(len(users_np) * percent / 2)
                edge_idx = [i for i in range(len(users_np))]
                drop_idx = random.sample(edge_idx, add_drop_num)
                user_np = []
                item_np = []
                for i in range(0, len(users_np)):
                    if i not in drop_idx:
                        user_np.append(users_np[i])
                        item_np.append(items_np[i])
                l = [(i, j) for i in range(self.num_users) for j in range(self.num_items)]
                add_list = random.sample(l, add_drop_num)
                for i in add_list:
                    user_np.append(i[0])
                    item_np.append(i[1])
                user_np = np.array(user_np)
                item_np = np.array(item_np)
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_users)),
                                        shape=(n_nodes, n_nodes))

            if aug_type == 'GCA':
                edge_weights = self.degree_drop_weights(users_items)
                edge_weights = edge_weights / edge_weights.mean() * self.p
                edge_weights = edge_weights.where(edge_weights < self.threshold,
                                                  torch.ones_like(edge_weights) * self.threshold)
                sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)
                user_np = []
                item_np = []
                for i in range(0, len(users_np)):
                    if not sel_mask[i]:
                        user_np.append(users_np[i])
                        item_np.append(items_np[i])
                user_np = np.array(user_np)
                item_np = np.array(item_np)
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np  + self.num_users)),
                                        shape=(n_nodes, n_nodes))
            if aug_type == 'ProGNN':
                user_np = []
                item_np = []
                for user in range(self.num_users):
                    for item in range(self.num_items):
                        if self.eadj.estimated_adj[user][item]>self.thre:
                            user_np.append(user)
                            item_np.append(item)
                user_np = np.array(user_np)
                item_np = np.array(item_np)
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np  + self.num_users)),
                                        shape=(n_nodes, n_nodes))



        else:
            ratings = np.ones_like(users_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (users_np, items_np+self.num_users)), shape=(n_nodes, n_nodes))
            if self.aug_type=='ProGNN':
                self.adj = torch.from_numpy(sp.csr_matrix((ratings, (users_np, items_np)),
                                                          shape=(self.num_users, self.num_items)).todense()).to(
                    self.device)
        adj_mat = tmp_adj + tmp_adj.T

        # normalize adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        return adj_matrix

    def train_model(self):
        data_iter = PairwiseSamplerV2(self.dataset.train_data, num_neg=1, batch_size=self.batch_size, shuffle=True)                    
        self.logger.info(self.evaluator.metrics_info())
        stopping_step = 0
        com_rel = np.loadtxt(self.load_dir + "/Com_relation.txt", dtype='float32')
        com_rel = (com_rel - com_rel.min()) / (com_rel.max() - com_rel.min())
        uc = np.loadtxt(self.load_dir + "/User_com.txt", dtype='float32')
        ic = np.loadtxt(self.load_dir + "/Item_com.txt", dtype='float32')
        com_graph = self.create_adj_mat(uc, ic, com_rel, is_subgraph=True, aug_type="com")
        com_graph = sp_mat_to_sp_tensor(com_graph).to(self.device)
        if self.aug_type=="Com":
            sub_graph1 = self.create_adj_mat(uc,ic,com_rel,is_subgraph=True,aug_type="node-e")
            sub_graph1 = sp_mat_to_sp_tensor(sub_graph1).to(self.device)
            sub_graph2 = self.create_adj_mat(uc,ic,com_rel,is_subgraph=True, aug_type="edge-e")
            sub_graph2 = sp_mat_to_sp_tensor(sub_graph2).to(self.device)
        else:
            sub_graph2 = None
            if self.aug_type is None:
                sub_graph1=None
            else:
                sub_graph1 = self.create_adj_mat(uc, ic, com_rel, is_subgraph=True, aug_type=self.aug_type)
                sub_graph1 = sp_mat_to_sp_tensor(sub_graph1).to(self.device)

        uc = torch.from_numpy(uc).to(self.device)
        ic = torch.from_numpy(ic).to(self.device)

        for epoch in range(1, self.epochs + 1):
            total_loss, total_bpr_loss, total_reg_loss, total_reg_loss1, total_reg_loss2 , total_reg_loss3 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            training_start_time = time()
            self.lightgcn.train()
            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                sup_logits= self.lightgcn(
                    sub_graph1, sub_graph2,uc,ic,com_graph, bat_users, bat_pos_items, bat_neg_items)
                
                # BPR Loss
                bpr_loss = -torch.sum(F.logsigmoid(sup_logits))

                # Reg Loss
                reg_loss = l2_loss(
                    self.lightgcn.user_embeddings(bat_users),
                    self.lightgcn.item_embeddings(bat_pos_items),
                    self.lightgcn.item_embeddings(bat_neg_items),
                )
                if self.aug_type=='ProGNN':
                    loss_fro = torch.norm(self.eadj.estimated_adj - self.adj, p='fro')
                    loss_l1 = torch.norm(self.eadj.estimated_adj, 1)
                    loss_nuclear = torch.norm(self.eadj.estimated_adj, p="nuc")

                loss = bpr_loss + self.reg * reg_loss
                if self.aug_type == 'ProGNN':
                    loss =loss+self.reg3 * loss_fro +self.reg1 * loss_l1+ self.reg2 * loss_nuclear
                total_loss += loss
                total_bpr_loss += bpr_loss
                total_reg_loss += self.reg * reg_loss
                if self.aug_type == 'ProGNN':
                    total_reg_loss3+= self.reg3 * loss_fro
                    total_reg_loss1 += self.reg1 * loss_l1
                    total_reg_loss2 += self.reg2 * loss_nuclear
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if self.aug_type == 'ProGNN':
                self.logger.info("[iter %d : loss : %.4f = %.4f + %.4f+ %.4f + %.4f+ %.4f  , time: %f]" % (
                    epoch,
                    total_loss / self.num_ratings,
                    total_bpr_loss / self.num_ratings,
                    total_reg_loss / self.num_ratings,
                    total_reg_loss3 / self.num_ratings,
                    total_reg_loss1 / self.num_ratings,
                    total_reg_loss2 / self.num_ratings,
                    time() - training_start_time,))
            else:
                self.logger.info("[iter %d : loss : %.4f = %.4f + %.4f  , time: %f]" % (
                    epoch,
                    total_loss/self.num_ratings,
                    total_bpr_loss / self.num_ratings,
                    total_reg_loss / self.num_ratings,
                    # total_reg_loss3 / self.num_ratings,
                    # total_reg_loss1 / self.num_ratings,
                    # total_reg_loss2 / self.num_ratings,
                    time()-training_start_time,))

            if epoch % self.verbose == 0 and epoch > self.config['start_testing_epoch']:
                result, flag = self.evaluate_model(sub_graph1, sub_graph2, uc,ic,com_graph)
                self.logger.info("epoch %d:\t%s" % (epoch, result))
                if flag:
                    self.best_epoch = epoch
                    stopping_step = 0
                    self.logger.info("Find a better model.")
                    if self.save_flag:
                        self.logger.info("Save model to file as pretrain.")
                        torch.save(self.lightgcn.state_dict(), self.tmp_model_dir)
                        self.saver.save(self.sess, self.tmp_model_dir)
                else:
                    stopping_step += 1
                    if stopping_step >= self.stop_cnt:
                        self.logger.info("Early stopping is trigger at epoch: {}".format(epoch))
                        break

        self.logger.info("best_result@epoch %d:\n" % self.best_epoch)
        nni.report_final_result(self.best_result[1])
        if self.save_flag:
            self.logger.info('Loading from the saved best model during the training process.')
            self.lightgcn.load_state_dict(torch.load(self.tmp_model_dir))
            uebd = self.lightgcn.user_embeddings.weight.cpu().detach().numpy()
            iebd = self.lightgcn.item_embeddings.weight.cpu().detach().numpy()
            np.save(self.save_dir + 'user_embeddings.npy', uebd)
            np.save(self.save_dir + 'item_embeddings.npy', iebd)
            buf, _ = self.evaluate_model(sub_graph1, sub_graph2, uc,ic,com_graph)
        elif self.pretrain_flag:
            buf, _ = self.evaluate_model(sub_graph1, sub_graph2, uc,ic,com_graph)
        else:
            buf = '\t'.join([("%.4f" % x).ljust(12) for x in self.best_result])
        self.logger.info("\t\t%s" % buf)

    # @timer
    def evaluate_model(self,sub_graph1, sub_graph2, uc,ic,com_graph):
        flag = False
        self.lightgcn.eval(sub_graph1, sub_graph2, uc,ic,com_graph)
        current_result, buf = self.evaluator.evaluate(self)
        nni.report_intermediate_result(current_result[1])
        if self.best_result[1] < current_result[1]:
            self.best_result = current_result
            flag = True
        return buf, flag

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        return self.lightgcn.predict(users).cpu().detach().numpy()
