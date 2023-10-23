import copy
import math

import numpy as np
import torch
from tqdm import tqdm

from proc.model import ModelHandler
from cfg import Config as cfg

def load2cuda(batch):
    assert (len(batch) == 2 or len(batch) == 3)
    _data, data = batch[0], batch[1]
    gid, gt, gl, tl, nm = data
    gid, gt, gl, tl = gid.to(cfg.device), gt.to(cfg.device), gl.to(cfg.device), tl.to(cfg.device)
    if len(batch) == 3:  # train
        l = batch[2]
        l = l.to(cfg.device)
        nm = nm.to(cfg.device)
        return _data, (gid, gt, gl, tl, nm), l
    return _data, (gid, gt, gl, tl, nm)  # valid

def run_valid(valid_loader, model: ModelHandler, dist_metrix, valid_range, epoch=math.nan, quick=True):
    model.eval()
    traj_embed_list = []
    with torch.no_grad():
        with tqdm(valid_loader, str(epoch + 1) + ">epoch valid:") as tq:
            for i, batch in enumerate(tq):
                batch = load2cuda(batch)
                data, data2 = batch
                embed = model(data2, train=False, batch_size=cfg.valid_batch_size)
                traj_embed_list.extend(embed.squeeze())
    embedding = torch.stack(traj_embed_list)
    if quick:
        acc = quick_top10_acc(dist_metrix,embedding,valid_range)
        return acc,None,None
    acc = cal_top10_acc(dist_metrix, embedding, valid_range)
    return acc

def cal_top10_acc(ground_truth, traj_embeddings, _range):
    ground_truth = ground_truth[_range[0]:_range[-1] + 1, _range[0]:_range[-1] + 1]
    traj_embeddings = traj_embeddings[_range[0]:_range[-1] + 1]
    fake_metrix = torch.zeros((ground_truth.shape[0], ground_truth.shape[1]))

    for i in range(len(ground_truth)):
        traj_e = traj_embeddings[i].repeat(len(traj_embeddings), 1)
        fake_metrix[i, :] = torch.sum(torch.square(traj_e - traj_embeddings), dim=-1)

    acc_10 = calculate_top_K_accuracy_for_whole_valid_traj(ground_truth, fake_metrix.cpu().numpy())
    acc_50 = calculate_top_K_accuracy_for_whole_valid_traj(ground_truth, fake_metrix.cpu().numpy(), tops=50, topf=50)
    acc_10of50 = calculate_top_K_accuracy_for_whole_valid_traj(ground_truth, fake_metrix.cpu().numpy(), tops=10,
                                                               topf=50)
    return acc_10, acc_50, acc_10of50

def quick_top10_acc(ground_truth, traj_embeddings, _range):
    ground_truth = ground_truth[_range[0]:_range[-1] + 1, _range[0]:_range[-1] + 1]
    traj_embeddings = traj_embeddings[_range[0]:_range[-1] + 1]
    fake_metrix = torch.zeros((ground_truth.shape[0], ground_truth.shape[1]))

    for i in range(len(ground_truth)):
        traj_e = traj_embeddings[i].repeat(len(traj_embeddings), 1)
        fake_metrix[i, :] = torch.sum(torch.square(traj_e - traj_embeddings), dim=-1)

    acc_10 = calculate_top_K_accuracy_for_whole_valid_traj(ground_truth, fake_metrix.cpu().numpy())
    # acc_50 = calculate_top_K_accuracy_for_whole_valid_traj(ground_truth, fake_metrix.cpu().numpy(), tops=50, topf=50)
    # acc_10of50 = calculate_top_K_accuracy_for_whole_valid_traj(ground_truth, fake_metrix.cpu().numpy(), tops=10,
    return acc_10#, acc_50, acc_10of50

def calculate_top_K_accuracy_for_whole_valid_traj(sim_valid, fake_metrix, tops=10, topf=10):
    if len(sim_valid) != fake_metrix.shape[0] or len(sim_valid) != fake_metrix.shape[1]:
        raise Exception("similar metrix shape dismatch")
    if max(tops, topf) >= fake_metrix.shape[0]:
        raise Exception("too less traj(", fake_metrix.shape[0], ")for top", tops, topf)

    sum_acc = 0.0
    with tqdm(range(len(sim_valid)), "calculating nearist traj....") as tq:
        for i in tq:
            topA = calculate_top_K_min(sim_valid[i], tops, skip=i)
            topB = calculate_top_K_min(fake_metrix[i], topf, skip=i)
            if not (topf == len(topB) and len(topA) == tops):
                raise Exception(f"error topA B len,{len(topA)},{len(topB)}")
            common = [v for v in topA if v in topB]
            acc = float(len(common)) / tops
            sum_acc = sum_acc + acc
    return sum_acc / fake_metrix.shape[0]


def calculate_top_K_min(dist, topk, skip):
    ret = []
    dist = copy.deepcopy(dist)
    dist[skip] = np.Inf
    for i in range(topk):
        min = np.Inf
        pos = -1
        for j in range(len(dist)):
            if dist[j] < min:
                min = dist[j]
                pos = j
        if pos == -1:
            raise Exception("..." + str(dist))
            #exit(123)
        else:
            ret.append(pos)
            dist[pos] = np.Inf
    return ret
