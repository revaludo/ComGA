"""
Paper: Self-supervised Graph Learning for Recommendation
Author: Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, Jianxun Lian, and Xing Xie
Reference: https://github.com/wujcan/DisCom-Torch
"""

__author__ = "Jiancan Wu"
__email__ = "wujcan@gmail.com"

__all__ = ["DisCom"]

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


class _LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, norm_adj, n_layers):
        super(_LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
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

    def forward(self):
        user_embeddings, item_embeddings = self._forward_gcn(self.norm_adj)
        return  user_embeddings, item_embeddings

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

    def predict(self, users):
        if self._user_embeddings_final is None or self._item_embeddings_final is None:
            raise ValueError("Please first switch to 'eval' mode.")
        user_embs = F.embedding(users, self._user_embeddings_final)
        temp_item_embs = self._item_embeddings_final
        ratings = torch.matmul(user_embs, temp_item_embs.T)
        return ratings

    def eval(self):
        super(_LightGCN, self).eval()
        self._user_embeddings_final, self._item_embeddings_final = self._forward_gcn(self.norm_adj)

class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, u,i):
        super(EstimateAdj, self).__init__()
        self.estimated_adj = nn.Parameter(torch.FloatTensor(u,i))
    def forward(self):
        return self.estimated_adj
class DisCom(AbstractRecommender):
    def __init__(self, config):
        super(DisCom, self).__init__(config)

        self.config = config
        self.model_name = config["recommender"]
        self.dataset_name = config["dataset"]
        self.aug_type = config["aug_type"]
        # General hyper-parameters
        self.reg = config['reg']
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
        self.tmp_model_dir = None
        self.save_dir = config.data_dir + '%s/pretrain-embeddings/%s/%s/n_layers=%d/' % (
            self.dataset_name,
            self.model_name,
            self.aug_type,
            self.n_layers,)
        ensureDir(self.save_dir)
        if self.pretrain_flag:
            self.tmp_model_dir = config.data_dir + '%s/model_tmp/%s/%s/' % (
                self.dataset_name,
                self.model_name,
                self.model_str)+'best.pkl'
            ensureDir(self.tmp_model_dir)

        self.num_users, self.num_items, self.num_ratings = self.dataset.num_users, self.dataset.num_items, self.dataset.num_train_ratings

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        adj_matrix = self.create_adj_mat()

        adj_matrix = sp_mat_to_sp_tensor(adj_matrix).to(self.device)

        self.lightgcn = _LightGCN(self.num_users, self.num_items, self.emb_size,
                                  adj_matrix, self.n_layers).to(self.device)
        self.dis_com_u = nn.Linear(self.emb_size, self.user_com).to(self.device)
        self.dis_com_i = nn.Linear(self.emb_size, self.item_com).to(self.device)
        if self.pretrain_flag:
            self.lightgcn.reset_parameters(pretrain=self.pretrain_flag, dir=self.save_dir)
        else:
            self.lightgcn.reset_parameters(init_method=self.param_init)
        if self.aug_type=="Com":
            self.dis_com_r = nn.Linear(self.num_users + self.num_items, 1).to(self.device)
            self.optimizer = torch.optim.Adam(list(self.lightgcn.parameters())+list(self.dis_com_u.parameters())+list(self.dis_com_i.parameters())+list(self.dis_com_r.parameters()), lr=self.lr)
        if self.aug_type == 'PB':
            self.eadj = EstimateAdj(self.user_com, self.item_com).to(self.device)
            self.optimizer = torch.optim.Adam(list(self.lightgcn.parameters()) +list(self.dis_com_u.parameters())+list(self.dis_com_i.parameters())+ list(self.eadj.parameters()),
                                              lr=self.lr)
        if self.aug_type == 'SB':
            self.optimizer = torch.optim.Adam(
                list(self.lightgcn.parameters()) + list(self.dis_com_u.parameters()) + list(
                    self.dis_com_i.parameters()) ,
                lr=self.lr)
        # self.optimizer = torch.optim.Adam(list(self.lightgcn.parameters())+list(self.dis_com_u.parameters())+list(self.dis_com_i.parameters()), lr=self.lr)

        # self.com_rel = nn.Parameter(torch.rand(self.opt['user_com'], self.opt['item_com']))
        # self.optimizer = torch.optim.Adam(list(self.dis_com_u.parameters())+list(self.dis_com_i.parameters())+list(self.com_rel.parameters()), lr=self.lr)
    @timer
    def create_adj_mat(self, is_subgraph=False, aug_type='ed'):
        n_nodes = self.num_users + self.num_items
        users_items = self.dataset.train_data.to_user_item_pairs()
        users_np, items_np = users_items[:, 0], users_items[:, 1]
        ratings = np.ones_like(users_np, dtype=np.float32)
        if self.aug_type == 'SB':
            self.adj = torch.from_numpy(
                sp.csr_matrix((ratings, (users_np, items_np)), shape=(self.num_users, self.num_items)).todense()).to(
                self.device)
        tmp_adj = sp.csr_matrix((ratings, (users_np, items_np+self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # normalize adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        return adj_matrix
    # def sparse_dense_mul(self,s, d):
    #     i = s._indices()
    #     v = s._values()
    #     dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
    #     return torch.sparse.FloatTensor(i, v * dv, s.size())
    # def normalize(self, mx):
    #     rowsum = mx.sum(1)
    #     r_inv = rowsum.pow(-1 / 2).flatten()
    #     r_inv[torch.isinf(r_inv)] = 0.
    #     r_mat_inv = torch.diag(r_inv)
    #     mx = r_mat_inv @ mx
    #     mx = mx @ r_mat_inv
    #     return mx
    def train_model(self):
        data_iter = PairwiseSamplerV2(self.dataset.train_data, num_neg=1, batch_size=self.batch_size, shuffle=True)                    
        self.logger.info(self.evaluator.metrics_info())
        stopping_step = 0
        for epoch in range(1, self.epochs + 1):
            total_loss, total_bpr_loss, total_reg_loss = 0.0, 0.0, 0.0
            training_start_time = time()
            self.lightgcn.train()
            self.optimizer.zero_grad()
            ue, ie= self.lightgcn()
            self.uc = torch.softmax(self.dis_com_u(ue), dim=-1)
            self.ic = torch.softmax(self.dis_com_i(ie), dim=-1)
            com_user = self.uc.transpose(0, 1)
            com_item = self.ic.transpose(0, 1)

            if self.aug_type=='SB':
                self.com_rel = torch.zeros(self.user_com, self.item_com)
                for i in range(0, self.user_com):
                    users = com_user[i]
                    for j in range(0, self.item_com):
                        items = com_item[j]
                        self.com_rel[i][j] = torch.sum(
                            torch.mul(self.adj, torch.matmul(torch.unsqueeze(users, 1), torch.unsqueeze(items, 0))))
                self.com_rel = F.normalize(self.com_rel,p=1,dim=1).to(self.device)
            if self.aug_type=='PB':
                self.com_rel =self.eadj.estimated_adj
                self.com_rel =F.normalize(self.com_rel,p=1,dim=1).to(self.device)
            if self.aug_type == 'Com':
                uall = com_user.repeat(self.item_com, 1)
                iall = com_item.repeat(self.user_com, 1)
                self.com_rel = torch.sigmoid(self.dis_com_r(torch.cat((uall, iall), dim=1))).view(self.user_com,
                                                                                                  self.item_com)

            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                bue=F.embedding(bat_users, self.uc )
                bie=F.embedding(bat_pos_items, self.ic)
                bnie=F.embedding(bat_neg_items, self.ic)
                pos_One = torch.sigmoid(
                    torch.matmul(torch.matmul(bue, self.com_rel), bie.transpose(0, 1)))
                neg_One = torch.sigmoid(
                    torch.matmul(torch.matmul(bue, self.com_rel), bnie.transpose(0, 1)))
                # Label = torch.cat((torch.ones_like(pos_One), torch.zeros_like(neg_One)))
                # pre = torch.cat((pos_One, neg_One))
                sup_logits = pos_One - neg_One
                # BPR Loss
                bpr_loss = -torch.sum(F.logsigmoid(sup_logits))

                # Reg Loss
                reg_loss = l2_loss(
                    self.lightgcn.user_embeddings(bat_users),
                    self.lightgcn.item_embeddings(bat_pos_items),
                    self.lightgcn.item_embeddings(bat_neg_items),
                )

                # InfoNCE Loss

                loss = bpr_loss +  self.reg * reg_loss
                total_loss += loss
                total_bpr_loss += bpr_loss
                total_reg_loss += self.reg * reg_loss

            total_loss.backward()
            self.optimizer.step()

            self.logger.info("[iter %d : loss : %.4f = %.4f + %.4f, time: %f]" % (
                epoch, 
                total_loss/self.num_ratings,
                total_bpr_loss / self.num_ratings,
                total_reg_loss / self.num_ratings,
                time()-training_start_time,))

            if epoch % self.verbose == 0 and epoch > self.config['start_testing_epoch']:
                result, flag = self.evaluate_model()
                self.logger.info("epoch %d:\t%s" % (epoch, result))
                if flag:
                    self.best_epoch = epoch
                    stopping_step = 0
                    self.logger.info("Find a better model.")
                    if self.save_flag:
                        self.logger.info("Save model to file as pretrain.")
                        torch.save(self.lightgcn.state_dict(), self.tmp_model_dir)
                        self.saver.save(self.sess, self.tmp_model_dir)
                    uc = self.uc.cpu().detach().numpy()
                    ic = self.ic.cpu().detach().numpy()
                    uic=self.com_rel.cpu().detach().numpy()
                    np.savetxt(self.save_dir + 'User_com.txt', uc)
                    np.savetxt(self.save_dir + 'Item_com.txt', ic)
                    np.savetxt(self.save_dir + "Com_relation.txt", uic)
                else:
                    stopping_step += 1
                    if stopping_step >= self.stop_cnt:
                        self.logger.info("Early stopping is trigger at epoch: {}".format(epoch))
                        break
        nni.report_final_result(self.best_result[1])
        self.logger.info("best_result@epoch %d:\n" % self.best_epoch)
        if self.save_flag:
            self.logger.info('Loading from the saved best model during the training process.')
            self.lightgcn.load_state_dict(torch.load(self.tmp_model_dir))
            uebd = self.lightgcn.user_embeddings.weight.cpu().detach().numpy()
            iebd = self.lightgcn.item_embeddings.weight.cpu().detach().numpy()
            np.save(self.save_dir + 'user_embeddings.npy', uebd)
            np.save(self.save_dir + 'item_embeddings.npy', iebd)
            buf, _ = self.evaluate_model()
        elif self.pretrain_flag:
            buf, _ = self.evaluate_model()
        else:
            buf = '\t'.join([("%.4f" % x).ljust(12) for x in self.best_result])
        self.logger.info("\t\t%s" % buf)

    # @timer
    def evaluate_model(self):
        flag = False
        self.lightgcn.eval()
        current_result, buf = self.evaluator.evaluate(self)
        nni.report_intermediate_result(current_result[1])
        if self.best_result[1] < current_result[1]:
            self.best_result = current_result
            flag = True
        return buf, flag

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        return self.lightgcn.predict(users).cpu().detach().numpy()
