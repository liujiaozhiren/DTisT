import numpy as np
import torch
from torch import autograd, nn
from torch.nn import Parameter
import torch.nn.functional as F
from cfg import Config as cfg
from proc.info_nce import InfoNCE


class WeightMSELoss(torch.nn.Module):
    def __init__(self, batch_size, sampling_num=cfg.sampling_num):
        super(WeightMSELoss, self).__init__()
        self.weight = []
        for i in range(batch_size):
            tmp = []
            for traj_index in range(sampling_num):
                tmp.append(cfg.sampling_num - traj_index)
            self.weight.append(tmp)
        self.weight = np.array(self.weight, dtype=object).astype(np.float32)
        sum = np.sum(self.weight)
        self.weight = self.weight / sum
        self.weight = Parameter(torch.Tensor(self.weight).to(cfg.device), requires_grad=False)
        self.batch_size = batch_size
        self.sampling_num = sampling_num
        self.weight_sq = None

    def forward(self, input, target, isReLU=False, mask=None):
        div = target - input
        if isReLU:
            div = F.relu(-div)
        square = torch.mul(div, div)
        weight_square = torch.mul(square, self.weight)
        self.weight_sq = square
        if mask is not None:
            weight_square = weight_square * mask
        loss = torch.sum(weight_square)
        return loss

class Info_NCE(nn.Module):
    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.info_nce_loss = InfoNCE(temperature, reduction, negative_mode)
    def forward(self, traj_anchor, trajs_near, trajs_far):
        loss_info_nce = []
        for i in range(1, trajs_near.shape[1]):
            loss_info_nce.append(self.info_nce_loss(traj_anchor, trajs_near[:, i, :], trajs_far))
        return sum(loss_info_nce) / len(loss_info_nce)


class WeightedRankingLoss(torch.nn.Module):
    def __init__(self, batch_size=cfg.batch_size, sampling_num=cfg.sampling_num):
        super(WeightedRankingLoss, self).__init__()
        self.positive_loss = WeightMSELoss(batch_size, sampling_num)
        self.negative_loss = WeightMSELoss(batch_size, sampling_num)

    def forward(self, p_input, p_target, n_input, n_target, maskp=None, maskn=None):
        trajs_mse_loss = self.positive_loss(p_input, autograd.Variable(p_target).to(cfg.device), False, maskp)
        negative_mse_loss = self.negative_loss(n_input, autograd.Variable(n_target).to(cfg.device), True, maskn)
        loss = sum([trajs_mse_loss, negative_mse_loss])
        self.trajs_mse_loss = trajs_mse_loss.detach().cpu()
        self.negative_mse_loss = negative_mse_loss.detach().cpu()
        return loss

class EmbeddingLossMSE(torch.nn.Module):
    def __init__(self, batch_size=cfg.batch_size, sampling_num=cfg.sampling_num):
        super().__init__()
        self.weight = []
        for i in range(batch_size):
            tmp = []
            for traj_index in range(sampling_num):
                tmp.append(cfg.sampling_num - traj_index)
            self.weight.append(tmp)
        self.weight = np.array(self.weight, dtype=object).astype(np.float32)
        sum = np.sum(self.weight)
        self.weight = self.weight / sum
        self.weight = Parameter(torch.Tensor(self.weight).to(cfg.device), requires_grad=False)

    def forward(self, stu_anchor, stu_emb, tea_diff, emp, t_val= None, abs=True):
        if abs:
            stu_adj = torch.abs(stu_emb - stu_anchor)
            tea_diff = torch.abs(tea_diff)
        else:
            stu_adj = stu_emb - stu_anchor
        adj = F.pairwise_distance(stu_adj, tea_diff, p=2).view(emp.shape[0], emp.shape[1])
        adj = adj / adj.detach()
        if t_val is not None:
            adj = adj * t_val
        adj = adj * emp * self.weight
        return torch.sum(adj)

class EmbeddingLoss_CosSim(torch.nn.Module):
    def __init__(self, batch_size=cfg.batch_size, sampling_num=cfg.sampling_num):
        super().__init__()
        self.weight = []
        for i in range(batch_size):
            tmp = []
            for traj_index in range(sampling_num):
                tmp.append(cfg.sampling_num - traj_index)
            self.weight.append(tmp)
        self.weight = np.array(self.weight, dtype=object).astype(np.float32)
        sum = np.sum(self.weight)
        self.weight = self.weight / sum
        self.weight = Parameter(torch.Tensor(self.weight).to(cfg.device), requires_grad=False)

    def forward(self, stu_anchor, stu_emb, tea_diff, emp, t_val=None, abs=True):
        if abs:
            stu_adj = torch.abs(stu_emb - stu_anchor)
            tea_diff = torch.abs(tea_diff)
        else:
            stu_adj = stu_emb - stu_anchor
        adj = F.cosine_similarity(stu_adj, tea_diff, dim=-1).view(emp.shape[0], emp.shape[1])
        adj = adj
        if t_val is not None:
            adj = adj * t_val
        adj = adj * emp * self.weight
        return torch.sum(adj)



class Loss_stu(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = WeightedRankingLoss()
        self.loss2 = WeightedRankingLoss()
        self.pos, self.neg, self.tp, self.tn = 0, 0, 0, 0
        self.s, self.t = 0, 0
        self.info_nce = Info_NCE(negative_mode='paired')

    def forward(self, ret, target):
        stu_ret, tea_ret = ret
        traj_anchor, traj_other = stu_ret[:, 0, :], stu_ret[:, 1:, :]
        trajs_near, trajs_far = traj_other[:, :cfg.sampling_num, :], traj_other[:, cfg.sampling_num:, :]
        # shape = trajs_near.shape[0], trajs_near.shape[1]
        # trajs_anchor = traj_anchor.repeat((1, cfg.sampling_num, 1)).view(shape[1], shape[0], cfg.dim * 2)
        # trajs_anchor = trajs_anchor.transpose(0, 1).reshape(-1, cfg.dim * 2)
        # trajs_near_wl, trajs_far_wl = trajs_near.reshape(-1, cfg.dim * 2), trajs_far.reshape(-1, cfg.dim * 2)
        # pos_dist = F.pairwise_distance(trajs_anchor, trajs_near_wl, p=2).view(shape[0], shape[1])
        # neg_dist = F.pairwise_distance(trajs_anchor, trajs_far_wl, p=2).view(shape[0], shape[1])
        #
        # pos_target, neg_target = target[:, 0, :], target[:, 1, :]
        # pos_target, neg_target = torch.exp(-pos_target / cfg.max_sim * cfg.mail_pre_degree), \
        #     torch.exp(-neg_target / cfg.max_sim * cfg.mail_pre_degree)
        # pos_dist, neg_dist = torch.exp(-pos_dist), torch.exp(-neg_dist)
        # loss = self.loss(pos_dist, pos_target, neg_dist, neg_target)
        # self.pos, self.neg = self.loss.trajs_mse_loss, self.loss.negative_mse_loss
        loss = 0
        loss_info_nce = self.info_nce(traj_anchor, trajs_near, trajs_far)
        return loss + loss_info_nce

    def get_loss_info(self):
        return f"loss:{(self.s + self.t):.5f}(s {self.s:.5f}:{self.pos:.5f},{self.neg:.5f}|t {self.t:.3f}:{self.tp:.3f},{self.tn:.3f})"


class Loss_tea(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = WeightedRankingLoss()
        self.pos, self.neg = 0, 0
        self.info_nce = Info_NCE(negative_mode='paired')

    def forward(self, ret, target):
        _, tea_ret = ret

        pos_target, neg_target = target[:, 0, :], target[:, 1, :]
        pos_target, neg_target = torch.exp(-pos_target / cfg.max_sim * cfg.mail_pre_degree), \
            torch.exp(-neg_target / cfg.max_sim * cfg.mail_pre_degree)

        t_near = tea_ret[:, :cfg.sampling_num].reshape(-1, cfg.dim * 2)
        t_far = tea_ret[:, cfg.sampling_num:].reshape(-1, cfg.dim * 2)
        shape = tea_ret.shape[0], tea_ret.shape[1] // 2
        p_tea = F.pairwise_distance(t_near, torch.zeros_like(t_near, device=cfg.device), p=2).view(shape[0], shape[1])
        n_tea = F.pairwise_distance(t_far, torch.zeros_like(t_far, device=cfg.device), p=2).view(shape[0], shape[1])
        t_pos_dist, t_neg_dist = torch.exp(-p_tea), torch.exp(-n_tea)

        loss = self.loss(t_pos_dist, pos_target, t_neg_dist, neg_target)
        self.pos, self.neg = self.loss.trajs_mse_loss, self.loss.negative_mse_loss
        shape = tea_ret[:, :cfg.sampling_num].shape
        traj_anchor = torch.zeros((shape[0], shape[2]), device=cfg.device)
        trajs_near, trajs_far = tea_ret[:, :cfg.sampling_num], tea_ret[:, cfg.sampling_num:]
        if cfg.tea_nce:
            loss += self.info_nce(traj_anchor, trajs_near, trajs_far)
        return loss

    def get_loss_info(self):
        return f"tea loss:{(self.pos + self.neg):.5f}(p:{self.pos:.5f},n:{self.neg:.5f})"


class Loss_stu_mod(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = WeightedRankingLoss()
        self.loss_t_mse = EmbeddingLossMSE()
        self.loss_t_cossim = EmbeddingLoss_CosSim()
        self.pos_raw, self.neg_raw = 0, 0
        self.pos_mod, self.neg_mod = 0, 0
        self.info_nce = Info_NCE(negative_mode='paired')

    def forward(self, ret, target):
        stu_ret, tea_ret = ret
        traj_anchor, traj_other = stu_ret[:, 0, :], stu_ret[:, 1:, :]
        trajs_near, trajs_far = traj_other[:, :cfg.sampling_num, :], traj_other[:, cfg.sampling_num:, :]
        shape = trajs_near.shape[0], trajs_near.shape[1]
        trajs_anchor = traj_anchor.repeat((1, cfg.sampling_num, 1)).view(shape[1], shape[0], cfg.dim * 2)
        trajs_anchor = trajs_anchor.transpose(0, 1).reshape(-1, cfg.dim * 2)
        trajs_near, trajs_far = trajs_near.reshape(-1, cfg.dim * 2), trajs_far.reshape(-1, cfg.dim * 2)

        pos_dist = F.pairwise_distance(trajs_anchor, trajs_near, p=2).view(shape[0], shape[1])
        neg_dist = F.pairwise_distance(trajs_anchor, trajs_far, p=2).view(shape[0], shape[1])

        t_near = tea_ret[:, :cfg.sampling_num].reshape(-1, cfg.dim * 2).detach()
        t_far = tea_ret[:, cfg.sampling_num:].reshape(-1, cfg.dim * 2).detach()
        pos_dist_tea = F.pairwise_distance(t_near, torch.zeros_like(t_near, device=cfg.device), p=2).view(shape[0],
                                                                                                          shape[1])
        neg_dist_tea = F.pairwise_distance(t_far, torch.zeros_like(t_far, device=cfg.device), p=2).view(shape[0],
                                                                                                        shape[1])

        pos_target, neg_target = target[:, 0, :], target[:, 1, :]
        pos_target, neg_target = torch.exp(-pos_target / cfg.max_sim * cfg.mail_pre_degree), \
            torch.exp(-neg_target / cfg.max_sim * cfg.mail_pre_degree)
        pos_dist, neg_dist = torch.exp(-pos_dist), torch.exp(-neg_dist)
        pos_dist_tea, neg_dist_tea = torch.exp(-pos_dist_tea), torch.exp(-neg_dist_tea)
        pos_which = torch.abs(pos_dist - pos_target) < torch.abs(pos_dist_tea - pos_target)
        neg_which = torch.abs(neg_dist - neg_target) < torch.abs(neg_dist_tea - neg_target)
        loss = self.loss(pos_dist, pos_target, neg_dist, neg_target)
        loss_info_nce = self.info_nce(traj_anchor, trajs_near.view(shape[0], shape[1], cfg.dim * 2),
                                      trajs_far.view(shape[0], shape[1], cfg.dim * 2))
        loss += loss_info_nce

        pos_emp = torch.ones_like(pos_target)
        neg_emp = torch.ones_like(neg_target)
        pos_emp[pos_which] = 0.0
        neg_emp[neg_which] = 0.0
        p_dis_stu_target = self.loss.positive_loss.weight_sq
        _ = self.loss(pos_dist, pos_dist_tea, neg_dist, neg_dist_tea)
        p_dis_stu_tea = self.loss.positive_loss.weight_sq


        loss_mse_1_p = self.loss_t_mse(trajs_anchor, trajs_near, t_near, pos_emp, p_dis_stu_tea)
        loss_mse_2_p = self.loss_t_mse(trajs_anchor, trajs_near, t_near, pos_emp, p_dis_stu_target)
        loss_cos_1_p = self.loss_t_cossim(trajs_anchor, trajs_near, t_near, pos_emp, p_dis_stu_tea)
        loss_cos_2_p = self.loss_t_cossim(trajs_anchor, trajs_near, t_near, pos_emp, p_dis_stu_target)
        loss_mse_1_pf = self.loss_t_mse(trajs_anchor, trajs_near, t_near, pos_emp, p_dis_stu_tea, False)
        loss_mse_2_pf = self.loss_t_mse(trajs_anchor, trajs_near, t_near, pos_emp, p_dis_stu_target, False)
        loss_cos_1_pf = self.loss_t_cossim(trajs_anchor, trajs_near, t_near, pos_emp, p_dis_stu_tea, False)
        loss_cos_2_pf = self.loss_t_cossim(trajs_anchor, trajs_near, t_near, pos_emp, p_dis_stu_target, False)

        loss2 = loss_mse_2_p * cfg.tea_ratio +loss_cos_2_p * (1-cfg.tea_ratio)
        return 0.5*loss2 + 0.5*loss
