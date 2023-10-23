import numpy as np

from cfg import Config as cfg

def traj_clip_in_grid(trajs):
    grid_info_ = []
    # in: [traj] --> traj:[node] --> node:[x,y]
    trajs_in_grids = []
    for traj in trajs:
        grid_list = []
        last_grid = None
        gridx, gridy = get_grid(traj)
        traj[:, -2], traj[:, -1] = gridx, gridy
        for i in range(len(traj)):
            grid_idx = gridx[i] * cfg.grid_size[1] + gridy[i]
            if last_grid is None or last_grid != grid_idx:
                grid = (grid_idx, [])
                grid_list.append(grid)
            grid_list[-1][1].append(traj[i])
            last_grid = grid_idx
        cfg.max_grid_1traj = max(cfg.max_grid_1traj, len(grid_list))
        for i in range(len(grid_list)):
            grid_list[i] = (grid_list[i][0], np.array(grid_list[i][1]))
            grid_info_.append(len(grid_list[i][1]))
        trajs_in_grids.append(grid_list)

    grid_info_ = np.array(grid_info_)
    print("grid info mean", grid_info_.mean(), grid_info_.std(), grid_info_.min(), grid_info_.max())
    return trajs_in_grids

def get_grid(traj):
    minx, miny = cfg.coord_describe.x.min, cfg.coord_describe.y.min
    deltax, deltay = cfg.xgrid_accuracy, cfg.ygrid_accuracy
    gridx = ((traj[:, 0] - minx) / deltax).astype(np.int)
    gridy = ((traj[:, 1] - miny) / deltay).astype(np.int)
    return gridx, gridy