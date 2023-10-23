import random

import numpy as np
import torch

from cfg import Config as cfg
import argparse

def update( dic: dict):
    for k, v in dic.items():
        if hasattr(cfg, k):
            setattr(cfg,k, v)
def parse_args():
    # dont set default value here! -- it will incorrectly overwrite the values in config.py.
    # config.py is the correct place for default values.
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--sim_type', type=str, help='')
    parser.add_argument('--city', type=str, help='')
    parser.add_argument('--dim', type=int, help='')
    parser.add_argument('--lr_s', type=float, help='')
    parser.add_argument('--lr_d', type=float, help='')
    parser.add_argument('--lr_i', type=float, help='')
    parser.add_argument('--batch_size', type=int, help='')
    parser.add_argument('--xgrid_accuracy', type=float, help='')
    parser.add_argument('--ygrid_accuracy', type=float, help='')
    parser.add_argument('--sizex', type=int, help='')
    parser.add_argument('--sizey', type=int, help='')
    parser.add_argument('--tea_threshold', type=float, help='')
    parser.add_argument('--tea_ratio', type=float, help='')
    parser.add_argument('--layer_gru', type=int, help='')
    parser.add_argument('--layer_t', type=int, help='')
    parser.add_argument('--layer_M', type=int, help='')
    parser.add_argument('--layer_G', type=int, help='')
    parser.add_argument('--head1', type=int, help='')
    parser.add_argument('--head2', type=int, help='')
    parser.add_argument('--train_ratio', type=float, help='')
    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))

def dict_to_string(dictionary):
    result = ""
    for key, value in dictionary.items():
        result += "__" + str(key) + "_" + str(value)
    if len(dictionary)>0:
        result += "__"
    return result
def init_args():
    arg = parse_args()
    if 'layer_gru' in arg:
        arg['layer_t']= arg['layer_gru']
    if 'xgrid_accuracy'in arg:
        arg['ygrid_accuracy']= arg['xgrid_accuracy']
    update(arg)
    cfgUPD(arg)
def cfgUPD(d):
    cfg.extra_msg = dict_to_string(d)
    cfg.grid_size = [cfg.sizex, cfg.sizey]
    cfg.data_fn_prefix = 'data.' + cfg.city + '/'
    cfg.traj_data = cfg.data_fn_prefix + 'trajs.pkl'
    cfg.dist_data = cfg.data_fn_prefix + cfg.sim_type + '_dist.pkl'
    cfg.pp_data_fn = cfg.data_fn_prefix + cfg.sim_type + '_dist_pp.pkl'
    cfg.mod_filename = 'modfile/' + cfg.city + '.' + cfg.sim_type + '.' + cfg.extra_msg + "mod.file"


def seed_set(seed=0):
    if cfg.device != 'cpu':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)



