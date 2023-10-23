import numpy as np
import traj_dist.distance as tdist
from cfg import Config as cfg

def cal_node_pair(traj_1, traj_2, type="dtw", type_d="spherical"):
    traj_1 = np.array(traj_1)
    traj_2 = np.array(traj_2)
    dim_1 = traj_1.shape[1]
    dim_2 = traj_2.shape[1]

    if dim_1 != 2 or dim_2 != 2:
        raise ValueError("Trajectories should be in 2D. t1 is %dD and t2 is %d given" % (dim_1, dim_2))

    if not (type_d in ["spherical", "euclidean"]):
        raise ValueError("The type_d argument should be 'euclidean' or 'spherical'\ntype_d given is : " + type_d)

    # dtw_ is rewrite dtw func getting node pair. dist:(value,(traj1_pair_nodes,traj2_pair_nodes))
    if type == 'dtw':
        dist_func = tdist.METRIC_DIC[type_d]["dtw_"]
        dist = dist_func(traj_1, traj_2)
    elif type == 'hausdorff':
        dist_func = tdist.METRIC_DIC[type_d]["hausdorff_"]
        dist = dist_func(traj_1, traj_2)
    elif type == 'erp':
        dist_func = tdist.METRIC_DIC[type_d]["erp_"]
        g = np.array([cfg.coord_describe.x.mean, cfg.coord_describe.y.mean])
        dist = dist_func(traj_1, traj_2, g)
    elif type == 'frechet':
        dist_func = tdist.METRIC_DIC[type_d]["frechet_"]
        dist = dist_func(traj_1, traj_2)
    elif type.startswith('lcss'):
        dist_func = tdist.METRIC_DIC[type_d]["lcss_"]
        eps = cfg.eps_lcss
        dist = dist_func(traj_1, traj_2, eps)
    elif type.startswith('edr'):
        dist_func = tdist.METRIC_DIC[type_d]["edr_"]
        eps = cfg.eps_edr
        dist = dist_func(traj_1, traj_2, eps)
    else:
        raise Exception(type + "not support!")
    return dist
