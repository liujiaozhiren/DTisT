import copy
import os
import pickle

import numpy as np
from tqdm import tqdm
from cfg import Config as cfg, DotDict
from proc.data_construct import make_near_train_data, make_valid_data
from util.gird import traj_clip_in_grid

from util.sim_cal import cal_node_pair


def drop_duplicated_node(traj):
    last_node = None
    target_traj = []
    for node in traj:
        if last_node is not None and last_node[0] == node[0] and last_node[1] == node[1]:
            continue
        target_traj.append(node)
        last_node = node
    return np.array(target_traj)

def pre_load_for_pp(whole_data):
    for i, trajs in enumerate(whole_data):
        whole_data[i] = drop_duplicated_node(trajs)
    node_pair_metrix = np.zeros((len(whole_data), len(whole_data))).tolist()
    if os.path.exists(cfg.pp_data_fn):
        node_pair_metrix = pickle.load(open(cfg.pp_data_fn, 'rb'))
    else:
        with tqdm(range(len(whole_data)), "creat tmp np file", mininterval=5) as tq:
            for traj1_idx in tq:
                for traj2_idx in range(len(whole_data)):
                    node_pair_metrix[traj1_idx][traj2_idx] = None
    cfg.whole_trajs=whole_data
    return node_pair_metrix


def pre_proc(whole_data, sim_metrix, pp_metrix, train_range, valid_range, test_range):
    traj_describe(whole_data)
    enhanced_data = trajs_enhanced(whole_data)
    whole_grid_traj = traj_clip_in_grid(enhanced_data)
    train_data = make_near_train_data(enhanced_data, sim_metrix, pp_metrix, train_range, whole_grid_traj)
    valid_data = make_valid_data(enhanced_data, valid_range, whole_grid_traj)
    return train_data, valid_data

def traj_describe(data):
    x, y, l = [], [], []
    for traj in data:
        l.append(len(traj))
        for r in traj:
            x.append(r[0])
            y.append(r[1])
    coord_describe = {
        'x': {'mean': np.mean(x), 'std': np.std(x), 'min': np.min(x), 'max': np.max(x)},
        'y': {'mean': np.mean(y), 'std': np.std(y), 'min': np.min(y), 'max': np.max(y)},
        'l': {'mean': np.mean(l), 'std': np.std(l), 'min': np.min(l), 'max': np.max(l)}
    }

    coord_describe = DotDict(coord_describe)
    print("please rewrite cfg file, with coord_describe = ", coord_describe)
    cfg.coord_describe = coord_describe


def trajs_enhanced(trajs):
    ret = []
    for idx in range(len(trajs)):
        traj = drop_duplicated_node(trajs[idx])
        norm = norm_feat(traj)
        item = np.empty((len(traj), cfg.in_dim+2)) # + gridx gridy
        item[:, 0:2] = norm
        ret.append(item)
    return ret

def norm_feat(traj):
    cd = cfg.coord_describe
    meanx, meany, stdx, stdy = cd.x.mean, cd.y.mean, cd.x.std, cd.y.std
    traj = copy.deepcopy(traj)
    traj[:, 0] = (traj[:, 0] - meanx) / stdx
    traj[:, 1] = (traj[:, 1] - meany) / stdy
    return traj


