# This is a sample Python script.
import logging
import os.path
import pickle
from datetime import datetime

import numpy
import torch
from tqdm import tqdm

from cfg import Config as cfg
from proc.data_construct import collate_fn_no_clip_gru, collate_valid_fn_no_clip_gru
from proc.loss import Loss_stu, Loss_tea, Loss_stu_mod
from proc.model import ModelHandler
from proc.proctool import load2cuda, run_valid
from util.init import init_args, seed_set
from proc.preproc import pre_load_for_pp, pre_proc
from torch.utils.data import DataLoader



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    init_args()
    logging.basicConfig(level = logging.INFO,
            format = "[%(filename)s:%(lineno)s %(funcName)s()]%(asctime)s -> %(message)s",
            handlers = [logging.FileHandler(cfg.log_file+str(datetime.now().strftime("--%Y-%m-%d-%H:%M:%S")), mode = 'w'),
                        logging.StreamHandler()]
            )
    torch.multiprocessing.set_start_method("spawn")
    seed_set(cfg.seed)
    whole_data = pickle.load(open(cfg.traj_data, 'rb'))
    print("total ", len(whole_data), "trajectorys")
    sim_metrix = pickle.load(open(cfg.dist_data, 'rb'))
    if type(sim_metrix) != numpy.ndarray:
        sim_metrix = numpy.array(sim_metrix, dtype=numpy.float64)
    cfg.max_sim = sim_metrix.max()
    print("max sim:", cfg.max_sim)
    pp_metrix = pre_load_for_pp(whole_data)

    train_range, valid_range, test_range = [0, int(0.6 * len(whole_data))], [int(0.6 * len(whole_data)),
                                                                             int(0.8 * len(whole_data))], [
        int(0.8 * len(whole_data)), len(whole_data)]

    train_data, valid_data = pre_proc(whole_data, sim_metrix, pp_metrix, train_range, valid_range, test_range)
    pickle.dump(pp_metrix, open(cfg.pp_data_fn, 'wb'))
    print("finish np file")

    train_loader = DataLoader(train_data[:int(len(train_data) * cfg.train_ratio)], shuffle=False,
                              batch_size=cfg.batch_size,
                              collate_fn=collate_fn_no_clip_gru, num_workers=0)
    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=cfg.valid_batch_size,
                              collate_fn=collate_valid_fn_no_clip_gru)

    model = ModelHandler()
    if os.path.exists(cfg.mod_filename):
        logging.info("===loading model file===")
        model.load_state_dict(torch.load(cfg.mod_filename))
    opt_stu = torch.optim.Adam(model.stu.parameters(), lr=cfg.lr_s)
    opt_tea = torch.optim.Adam(model.tea.parameters(), lr=cfg.lr_d)
    opt_stu_mod = torch.optim.Adam(model.stu.parameters(), lr=cfg.lr_i)
    acc = 0
    # acc = run_valid(valid_loader, stu_model, sim_metrix, valid_range)
    print("raw acc", acc)
    bad_epoch, best_acc = 0, acc
    best_50, best_10_50 = 0, 0
    loss_s = Loss_stu()
    loss_t = Loss_tea()
    loss_m = Loss_stu_mod()
    model = model.to(cfg.device)
    valid_acc = 0
    epoch_ = 0
    total_step = 0
    logging.info(f"mod file ={cfg.mod_filename}")
    logging.info("start training========================================logging===================== ")
    if cfg.pre_train:
        logging.info("start pre-train========================================logging===================== ")
        for epoch in range(500):
            total_step += 1
            epoch_ = epoch + 1
            model.train()
            sum_loss, m_sum_l, t_sum_l, s_sum_l = 0, 0, 0, 0
            with tqdm(train_loader, str(epoch + 1) + ">epoch train: raw stu", mininterval=2) as tq:
                for i, batch in enumerate(tq):
                    batch = load2cuda(batch)
                    _, data2, labels = batch
                    embed = model(data=data2)
                    loss = loss_s(embed, labels)
                    opt_stu.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.stu.parameters(), max_norm=20, norm_type=2,
                                                       error_if_nonfinite=True)
                    opt_stu.step()
                    s_sum_l = s_sum_l + loss.detach().item()
            # print("epoch(", epoch + 1, ") train loss=", s_sum_l / len(train_loader))
            valid_acc, _, _ = run_valid(valid_loader, model, sim_metrix, valid_range)
            if valid_acc <= best_acc:
                bad_epoch += 1
                if bad_epoch > cfg.early_stop:
                    break
                print("in epoch(", epoch + 1, ") worse epoch(", bad_epoch, ") valid acc model(", valid_acc, ")")
                logging.info(f"epoch({epoch + 1}) worse pre-train loss={s_sum_l / len(train_loader)},acc={valid_acc}")
            else:
                best_acc = valid_acc
                bad_epoch = 0
                torch.save(model.state_dict(), cfg.mod_filename)
                print("in epoch(", epoch + 1, ")save best stu valid acc model(", valid_acc, ")")
                logging.info(f"epoch({epoch + 1}) better pre-train loss={s_sum_l / len(train_loader)},acc={valid_acc}")

    model.load_state_dict(torch.load(cfg.mod_filename))
    s_sum_l = 0
    logging.info(f"load mod file===={cfg.mod_filename}====================================logging===================== ")
    model.eval()
    with tqdm(train_loader, "checkpoint>epoch train: raw stu", mininterval=2) as tq:
        for i, batch in enumerate(tq):
            batch = load2cuda(batch)
            _, data2, labels = batch
            embed = model(data=data2)
            loss = loss_s(embed, labels)
            s_sum_l = s_sum_l + loss.detach().item()
    best_acc, b_ac50, b_ac10at50 = run_valid(valid_loader, model, sim_metrix, valid_range, quick=False)
    print(
        f"=================finish stu train=======best_acc10:{best_acc},50:{b_ac50},10@50:{b_ac10at50},epoch={epoch_}")
    logging.info(
        f"=================finish stu train=======best_acc10:{best_acc},50:{b_ac50},10@50:{b_ac10at50},epoch={epoch_}")
    teach_bad_epoch, _teach_bad_epoch = 0, 0
    best_acc_ = best_acc
    if cfg.tea_guide:
        _teach_bad_epoch = 0
        best_tea_loss = 1e10
        epoch_2 = 0
        while _teach_bad_epoch < cfg.tea_early_stop:
            epoch_2 += 1
            model.train()
            for step in range(50):
                t_sum_l = 0
                with tqdm(train_loader, str(epoch_2 + 1) + ">epoch train: fine-tune tea", mininterval=2) as tq:
                    for i, batch in enumerate(tq):
                        batch = load2cuda(batch)
                        _, data2, labels = batch
                        embed = model(data=data2)
                        loss = loss_t(embed, labels)
                        opt_tea.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.tea.parameters(), max_norm=20, norm_type=2,
                                                           error_if_nonfinite=True)
                        opt_tea.step()
                        t_sum_l = t_sum_l + loss.detach().item()
                tl_avg, sl_avg = t_sum_l / len(train_loader), s_sum_l / len(train_loader)
                print(
                    f"epoch{epoch_2 + 1} step:{step} tea loss:{tl_avg} stu:{sl_avg}")
                if tl_avg < sl_avg * cfg.tea_threshold and best_tea_loss >= tl_avg:
                    best_tea_loss = tl_avg
                    torch.save(model.state_dict(), cfg.mod_filename)
                    logging.info(
                        f"finish dual-train epoch{epoch_}/{epoch_2 + 1} (totalstep{total_step}) step:{step} tea loss:{tl_avg} stu:{sl_avg}")
                    break
            model.load_state_dict(torch.load(cfg.mod_filename))
            teach_bad_epoch = 0
            for step in range(50):
                total_step += 1
                m_sum_l, sum_loss = 0, 0
                with tqdm(train_loader, str(epoch_2 + 1) + ">epoch" + str(step) + "train: mod stu",
                          mininterval=2) as tq:
                    for i, batch in enumerate(tq):
                        batch = load2cuda(batch)
                        _, data2, labels = batch
                        embed = model(data=data2)
                        loss = loss_m(embed, labels)
                        opt_stu_mod.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.stu.parameters(), max_norm=20, norm_type=2,
                                                           error_if_nonfinite=True)
                        opt_stu_mod.step()
                        m_sum_l = m_sum_l + loss.detach().item()
                        loss2 = loss_s(embed, labels)
                        sum_loss = sum_loss + loss2.detach().item()
                ml_avg, m_sl_avg = m_sum_l / len(train_loader), sum_loss / len(train_loader)
                valid_acc, _, _ = run_valid(valid_loader, model, sim_metrix, valid_range)
                model.train()
                if valid_acc > best_acc_:
                    best_acc_ = valid_acc
                    teach_bad_epoch = 0
                    torch.save(model.state_dict(), cfg.mod_filename)
                    _teach_bad_epoch = 0
                    print(
                        f"epoch{epoch_}/{epoch_2 + 1} step:{step} save best mod loss:{ml_avg} stu:{m_sl_avg},acc:({valid_acc})")
                    logging.info(
                        f"epoch{epoch_}/{epoch_2 + 1} step:{step} (totalstep{total_step}) save best mod loss:{ml_avg} stu:{m_sl_avg},acc:({valid_acc})")
                else:
                    teach_bad_epoch += 1
                    print(
                        f"bad epoch{teach_bad_epoch}/{_teach_bad_epoch} mod loss:{ml_avg} stu:{m_sl_avg},acc:({valid_acc})")
                    logging.info(
                        f"epoch{epoch_}/{epoch_2 + 1} step:{step} (totalstep{total_step}) bad epoch{teach_bad_epoch}/{_teach_bad_epoch} mod loss:{ml_avg} stu:{m_sl_avg},acc:({valid_acc})")
                if teach_bad_epoch >= cfg.tea_early_stop:
                    _teach_bad_epoch += 1
                    break
            model.load_state_dict(torch.load(cfg.mod_filename))
        model.load_state_dict(torch.load(cfg.mod_filename))
        testacc = run_valid(valid_loader, model, sim_metrix, test_range, quick=False)

        print("best_acc is ", testacc)
        logging.info(f"best acc is {testacc} total{total_step}, epoch{epoch_}/{epoch_2}, ")

