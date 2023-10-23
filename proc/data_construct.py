import copy
import logging

import numpy as np
import torch
from tqdm import tqdm

from cfg import Config as cfg
from util.sim_cal import cal_node_pair


#from proc.preproc import lazy_index_list_select4list


def distance_sort_select(distance, index):
    dis = copy.deepcopy(np.array(distance[index]))
    index_dis_idx = np.argpartition(dis, cfg.sampling_num)[:cfg.sampling_num]
    b_sort_idx = np.argsort(dis[index_dis_idx])
    return index_dis_idx[b_sort_idx]

def negative_distance_sort_select(distance, index):
    dis = -copy.deepcopy(np.array(distance[index]))
    index_dis_idx = np.argpartition(dis, cfg.sampling_num_far)[:cfg.sampling_num_far]
    b_sort_idx = np.argsort(dis[index_dis_idx])
    return index_dis_idx[b_sort_idx]


def negative_distance_sample_sample_by_rate(distance, index, threshold=0):
    dis = copy.deepcopy(np.array(distance[index]))
    idxs = np.array([i for i, item in enumerate(dis) if item > threshold])

    item = np.array([item for i, item in enumerate(dis) if item > threshold])
    if len(idxs) < 10:
        idxs = np.array([i for i, item in enumerate(dis) if item >= threshold])
        item = np.array([item for i, item in enumerate(dis) if item >= threshold])
    item = np.log(item+1.)
    prob = item/np.sum(item)
    idxs = np.random.choice(idxs, size=cfg.sampling_num_far,replace=False,p=prob)
    return idxs

def negative_distance_sample_select(distance, index, threshold=0):
    dis = copy.deepcopy(np.array(distance[index]))
    idxs = np.random.choice(range(0, len(dis)), cfg.sampling_num_far * 2, replace=False)
    index_filtered = [item for item in idxs if dis[item] > threshold][:cfg.sampling_num_far]
    if len(index_filtered) < 10:
        index_filtered = [item for item in idxs if dis[item] >= threshold][:cfg.sampling_num_far]
    index_dis_idx = np.array(index_filtered)
    b_sort_idx = np.argsort(-dis[index_dis_idx])
    return index_dis_idx[b_sort_idx]

def make_near_train_data(trajs, sim_metrix, pp_metrix, train_range, whole_grid_traj):
    grid_traj = whole_grid_traj[train_range[0]:train_range[1]]
    trajs_train = trajs[train_range[0]:train_range[1]]
    sim_metrix = sim_metrix[train_range[0]:train_range[1], train_range[0]:train_range[1]]
    trajs_near_idxs, trajs_far_idxs = [], []
    if cfg.train_sample_func == 'sample_near_far_split':
        for idx, traj in enumerate(trajs_train):
            near_idxs = distance_sort_select(sim_metrix, idx)
            far_idxs = negative_distance_sort_select(sim_metrix, idx)
            trajs_near_idxs.append(near_idxs)
            trajs_far_idxs.append(far_idxs)
    elif cfg.train_sample_func == 'sorted_near_sample_far_split':
        for idx, traj in enumerate(trajs_train):
            near_idxs = distance_sort_select(sim_metrix, idx)
            if cfg.sim_type.startswith("lcss") or cfg.sim_type.startswith('edr'):
                far_idxs = negative_distance_sample_sample_by_rate(sim_metrix, idx,
                                                                  threshold=sim_metrix[idx, near_idxs[-1]])
            else:
                far_idxs = negative_distance_sample_select(sim_metrix, idx, threshold=sim_metrix[idx, near_idxs[-1]])
            trajs_near_idxs.append(near_idxs)
            trajs_far_idxs.append(far_idxs)
    elif cfg.train_sample_func == 'sorted_near_rate_far_split':
        for idx, traj in enumerate(trajs_train):
            near_idxs = distance_sort_select(sim_metrix, idx)
            far_idxs = negative_distance_sample_sample_by_rate(sim_metrix, idx, threshold=sim_metrix[idx, near_idxs[-1]])
            trajs_near_idxs.append(near_idxs)
            trajs_far_idxs.append(far_idxs)
    else:
        raise NotImplementedError
    train_set = []
    with tqdm(range(len(trajs_train)), " making train data...") as tq:
        for anchor_idx in tq:
            near_idxs = trajs_near_idxs[anchor_idx]
            far_idxs = trajs_far_idxs[anchor_idx]
            anchor_trajs = trajs_train[anchor_idx]
            near_trajs = index_list_select4list(trajs_train, near_idxs)
            far_trajs = index_list_select4list(trajs_train, far_idxs)
            anchor_grid_trajs = grid_traj[anchor_idx]
            near_grid_trajs = _lazy_index_list_select4list(grid_traj, near_idxs)
            far_grid_trajs = _lazy_index_list_select4list(grid_traj, far_idxs)
            near_labels = sim_metrix[anchor_idx][near_idxs]
            far_labels = sim_metrix[anchor_idx][far_idxs]
            near_pps = lazy_index_list_select4list(pp_metrix[anchor_idx], anchor_idx, near_idxs, sim_metrix[anchor_idx])
            far_pps = lazy_index_list_select4list(pp_metrix[anchor_idx], anchor_idx, far_idxs, sim_metrix[anchor_idx])
            near_item = (anchor_trajs, near_trajs, near_labels, near_pps, anchor_grid_trajs, near_grid_trajs)
            far_item = (anchor_trajs, far_trajs, far_labels, far_pps, anchor_grid_trajs, far_grid_trajs)
            item = (near_item, far_item)
            train_set.append(item)
    return train_set

def make_valid_data(trajs, _range, grid_traj):
    # grid_traj = whole_grid_traj[_range[0]:_range[1]]
    # trajs = trajs[_range[0]:_range[1]]
    target_set = []
    for traj_idx in range(len(trajs)):
        traj = trajs[traj_idx]
        grid_ = grid_traj[traj_idx]
        item = (traj, grid_)
        target_set.append(item)
    return target_set

def index_list_select4list(source_list, idx_list):
    target_list = []
    for idx in idx_list:
        target_list.append(np.array(source_list[idx]))
    return target_list

def _lazy_index_list_select4list(source_list, idx_list):
    target_list = []
    for idx in idx_list:
        target_list.append(source_list[idx])
    return target_list

def lazy_index_list_select4list(source_list, anchor_idx, idx_list,sim_list):
    target_list = []
    for idx in idx_list:
        if source_list[idx] is None: # lazy calculate point pair
            #logging.error("please check the point pair msg in traj_dist func")
            #raise Exception("please chk the traj_dist be modified support point-pair output")
            traj1, traj2 = cfg.whole_trajs[anchor_idx], cfg.whole_trajs[idx]
            sim_type = cfg.sim_type
            sim, node_pair = cal_node_pair(traj1, traj2, type=sim_type)
            if abs(sim-sim_list[idx]) > 1e-8:
                print("sim not equal", anchor_idx, idx, sim, sim_list[idx])
                raise Exception("sim not equal")

            if cfg.sim_type == 'frechet':
                x, y=[],[]
                for x_,y_ in node_pair:
                    x.append(x_)
                    y.append(y_)
                node_pair = x,y
            u, v = node_pair
            source_list[idx] = np.zeros((len(u), 2), dtype=np.long)
            source_list[idx][:, 0] = u
            source_list[idx][:, 1] = v
        target_list.append(source_list[idx])
    return target_list

def collate_fn_no_clip_gru(data):
    traj_max_len = 0
    label = torch.zeros((len(data), 2, cfg.sampling_num))
    all_trajs_grid = []
    all_trajs = []
    traj_max_grid = 0
    pp_infos = []
    # with static.t3[0]:
    for _i, batch in enumerate(data):
        near, far = batch  # near: anchor_trajs, near_trajs, near_labels, near_pps, anchor_grid_trajs, near_grid_trajs
        label[_i, 0, :] = torch.tensor(near[2])
        label[_i, 1, :] = torch.tensor(far[2])

        anchor_data = near[4]  # [grid]: grid: (grid_id, traj, grid_ret)
        near_datas = near[5]  # (anchor_trajs, near_trajs, near_labels, near_pps, anchor_grid_trajs, near_grid_trajs)
        far_datas = far[5]
        near_pp = near[3]
        far_pp = far[3]
        _pp = []
        _pp.extend(near_pp)
        _pp.extend(far_pp)
        pp_infos.append(_pp)
        all_trajs_grid.append(anchor_data)
        all_trajs_grid.extend(near_datas)
        all_trajs_grid.extend(far_datas)

        anchor_traj = near[0]
        near_trajs = near[1]
        far_trajs = far[1]
        all_trajs.append(anchor_traj)
        all_trajs.extend(near_trajs)
        all_trajs.extend(far_trajs)

        dim = anchor_traj.shape[-1]
    for i in range(len(all_trajs_grid)):
        traj_max_grid = max(len(all_trajs_grid[i]), traj_max_grid)
        traj_max_len = max(len(all_trajs[i]), traj_max_len)
    traj_grid_ids = np.zeros((len(all_trajs_grid), traj_max_grid))
    traj_grid_traj_len = np.zeros((len(all_trajs_grid), traj_max_grid))
    traj_grid_lens = np.zeros((len(all_trajs_grid)))
    traj_tensor = np.zeros((len(all_trajs_grid), traj_max_len, dim), dtype=np.float32)
    for i in range(len(all_trajs_grid)):
        traj = all_trajs_grid[i]
        traj_grid_lens[i] = len(traj)
        traj_tensor[i, :len(all_trajs[i])] = all_trajs[i]
        l = -1
        for j, grid in enumerate(traj):
            grid_id, grid_traj = grid
            traj_grid_ids[i, j] = grid_id
            l += len(grid_traj)
            traj_grid_traj_len[i, j] = l
    traj_grid_group_by_idxs = []
    for i, traj in enumerate(all_trajs):
        grid_cnt, k = 0, 0
        traj_tail_idx = traj_grid_traj_len[i]
        traj_grid_group_by_idx = []
        for j in range(len(all_trajs[i])):
            traj_grid_group_by_idx.append(grid_cnt)
            if j == traj_tail_idx[k]:
                grid_cnt += 1
                k += 1
        traj_grid_group_by_idxs.append(traj_grid_group_by_idx)
    traj_num = 0
    jpp_metrixs = np.zeros((len(pp_infos), len(pp_infos[0]), cfg.max_grid_1traj * 2, cfg.max_grid_1traj * 2))
    for i, pp_info in enumerate(pp_infos):
        traj_base_idx = traj_num
        base_traj_group_by_idx = traj_grid_group_by_idxs[traj_base_idx]
        traj_num += 1
        for j, pp in enumerate(pp_info):
            traj_idx = traj_num + j
            traj_group_by_idx = traj_grid_group_by_idxs[traj_idx]
            for item in pp:
                base_grp = base_traj_group_by_idx[item[0]]
                grp = traj_group_by_idx[item[1]]
                jpp_metrixs[i][j][cfg.max_grid_1traj + grp][base_grp] += 1
                jpp_metrixs[i][j][base_grp][cfg.max_grid_1traj + grp] += 1

        traj_num += len(pp_info)

    traj_grid_ids = load_data2tensor(traj_grid_ids, torch.long)
    traj_tensor = load_data2tensor(traj_tensor)
    traj_grid_traj_len = load_data2tensor(traj_grid_traj_len, torch.long)
    traj_grid_lens = load_data2tensor(traj_grid_lens, torch.long)
    label = load_data2tensor(label)
    jpp_metrixs = load_data2tensor(jpp_metrixs, torch.long)
    return data, (traj_grid_ids, traj_tensor, traj_grid_traj_len, traj_grid_lens, jpp_metrixs), label

def collate_valid_fn_no_clip_gru(data):
    traj_max_len = 0
    all_trajs_grid = []
    all_trajs = []
    traj_max_grid = 0
    # with static.t3[0]:
    for i, batch in enumerate(data):
        traj, grid_ = batch
        all_trajs_grid.append(grid_)
        all_trajs.append(traj)
        dim = traj.shape[-1]
        traj_max_grid = max(len(grid_), traj_max_grid)
        traj_max_len = max(len(traj), traj_max_len)
    traj_grid_ids = np.zeros((len(all_trajs_grid), traj_max_grid))
    traj_grid_traj_len = np.zeros((len(all_trajs_grid), traj_max_grid))
    traj_grid_lens = np.zeros((len(all_trajs_grid)))
    traj_tensor = np.zeros((len(all_trajs_grid), traj_max_len, dim), dtype=np.float32)
    for i in range(len(all_trajs_grid)):
        traj = all_trajs_grid[i]
        traj_grid_lens[i] = len(traj)
        traj_tensor[i, :len(all_trajs[i])] = all_trajs[i]
        l = -1
        for j, grid in enumerate(traj):
            grid_id, grid_traj = grid
            traj_grid_ids[i, j] = grid_id
            l += len(grid_traj)
            traj_grid_traj_len[i, j] = l

    traj_grid_ids = load_data2tensor(traj_grid_ids, torch.long)
    traj_tensor = load_data2tensor(traj_tensor)
    traj_grid_traj_len = load_data2tensor(traj_grid_traj_len, torch.long)
    traj_grid_lens = load_data2tensor(traj_grid_lens, torch.long)
    return data, (traj_grid_ids, traj_tensor, traj_grid_traj_len, traj_grid_lens, None)

def load_data2tensor(data, _type=None):
    if type(data) != torch.Tensor:
        data = torch.tensor(data)
    if _type != None:
        data = data.to(_type)
    return data.to('cpu')
