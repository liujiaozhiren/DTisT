import torch


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value


class Config:
    sim_type = 'erp'
    city = 'demo'
    dim: int = 128
    lr_s: float = 5e-5
    lr_d: float = 5e-5
    lr_i: float = 1e-5
    batch_size: int = 10
    tea_threshold = 0.2
    tea_ratio = 1
    layer_gru = 2  # 1
    layer_t = 2  # 1
    layer_M = 1
    layer_G = 2
    # head1 = 2
    head2 = 2
    train_ratio = 1.0

    sizex = 10000
    sizey = 10000

    grid_size = [sizex, sizey]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    seed = 10086
    data_fn_prefix = 'data.' + city + '/'
    traj_data = data_fn_prefix + 'trajs.pkl'
    dist_data = data_fn_prefix + sim_type + '_dist.pkl'
    pp_data_fn = data_fn_prefix + sim_type + '_dist_pp.pkl'
    extra_msg = ''
    mod_filename = 'modfile/' + city + '.' + sim_type + '.' + extra_msg + "mod.file"
    log_file = 'log/' + city + '.' + sim_type + '.' + extra_msg + ".log"

    valid_batch_size = 100
    sampling_num = 20
    sampling_num_far = 20
    # TODO: change by dataset | This is Porto.
    coord_describe = {
        'x': {'mean': -8.605918252722095, 'std': 0.07732929772314669, 'min': -13.172526, 'max': -6.843042},
        'y': {'mean': 41.177839592713646, 'std': 0.08629946008813362, 'min': 39.256344, 'max': 41.803092},
        'l': {'mean': 79.4058, 'std': 28.622475895002516, 'min': 25, 'max': 174}}
    xgrid_accuracy = 0.01
    ygrid_accuracy = 0.005
    # sim_describe = {'cnt': 303601, 'max': 4521227.398842962, 'min': 0.0, 'avg': 625745.415422993,
    #                 'std': 417773.4887675786}
    # sim_describe = DotDict(sim_describe)
    coord_describe = DotDict(coord_describe)

    max_sim = None
    in_dim = 2
    dropout = 0

    early_stop = 3
    tea_early_stop = 3
    train_sample_func = 'sorted_near_sample_far_split'  # 'sample_near_far_split' or 'sorted_near_sample_far_split'

    tea_nce = False

    pre_train = True
    tea_guide = True
    whole_trajs = None  # static var
    max_grid_1traj = 0  # static var
