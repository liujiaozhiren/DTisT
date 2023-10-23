import copy
import math
import os
from torch.nn import Transformer
import torch
from torch import nn

from cfg import Config as cfg

class ModelPackage(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

class MonotonicModule(nn.Module):
    def __init__(self, A=None):
        super(MonotonicModule, self).__init__()
        if A is None:
            A = torch.linspace(-50, 50, 300)
            A[0] = -1e8
        # 将数组 A 转换为可学习的参数
        self.A = nn.Parameter(A, ).to(torch.float32).to(cfg.device)

    def forward(self, input_tensor):
        # 将输入张量展平为一维
        input_flat = input_tensor.view(-1)
        input_flat[input_flat > 0] = 1
        # 使用 torch.index_select() 快速查找数组 A 中对应的值
        output_flat = torch.index_select(self.A, dim=0, index=input_flat)

        # 将输出张量重新恢复为原始形状
        output_tensor = output_flat.view(input_tensor.size()).float()

        return output_tensor
class PositionalEncoding(torch.nn.Module):
    def __init__(self, hidden_size, max_seq_len=500, batch_first=True):
        super(PositionalEncoding, self).__init__()
        self.batch_first = batch_first
        pe = torch.zeros(max_seq_len, hidden_size)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) if batch_first else pe.unsqueeze(1)
        self.register_buffer('pe', pe)
        self.weight = torch.ones([1], requires_grad=True, dtype=torch.float32, device=cfg.device)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)] * self.weight
        return x


class TrajGru(nn.Module):
    def __init__(self, in_dim, out_dim, layer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer = layer
        self.gru = nn.GRU(self.in_dim, self.out_dim, num_layers=self.layer, batch_first=True, dropout=cfg.dropout,
                          bidirectional=True).to(cfg.device)

    def forward(self, batch_grid_traj, hs=None):
        output, hs = self.gru(batch_grid_traj[:, :, 0:2])
        return output, hs

class TransformerEncoder(nn.Module):
    def __init__(self, feature_dim,num_layer, head=2,  batch_first=True, drop_out=cfg.dropout):
        super(TransformerEncoder, self).__init__()
        self.position_encoding = PositionalEncoding(hidden_size=feature_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=head, batch_first=batch_first,
                                                        dropout=drop_out)
        self.transf_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)

    def forward(self, batch_traj_feat, key_pad_mask=None, mask=None):
        batch_traj_feat = self.position_encoding(batch_traj_feat)
        batch_traj_feat = self.transf_encoder(batch_traj_feat, src_key_padding_mask=key_pad_mask, mask=mask)
        return batch_traj_feat

class TransformerEncoderNoPos(nn.Module):
    def __init__(self, feature_dim, head=2, num_layer=1, batch_first=True, drop_out=cfg.dropout):
        super(TransformerEncoderNoPos, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=head, batch_first=batch_first,
                                                        dropout=drop_out)
        self.transf_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)

    def forward(self, batch_traj_feat, key_pad_mask=None, mask=None):
        batch_traj_feat = self.transf_encoder(batch_traj_feat, src_key_padding_mask=key_pad_mask, mask=mask)
        return batch_traj_feat

class GRUTransformerClip(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gru = TrajGru(in_dim=cfg.in_dim, out_dim=cfg.dim, layer=cfg.layer_gru).to(cfg.device)
        self.transformer = TransformerEncoder(cfg.dim * 2, num_layer=cfg.layer_t).to(cfg.device)

    def forward(self, grid_batch_tensor, grid_batchs_traj_len, traj_grid_lens):

        output, grid_state = self.gru(grid_batch_tensor)

        traj_seq, pad_mask, attn_mask = self.gru_ret2transfseq(output, grid_batchs_traj_len, traj_grid_lens)

        trajs_node_emb = self.transformer(traj_seq, key_pad_mask=pad_mask, mask=attn_mask < 0)

        return trajs_node_emb

    def gru_ret2transfseq(self, output, grid_batchs_traj_len, traj_grid_lens):
        mask_out = []
        for b, v in enumerate(grid_batchs_traj_len):
            mask_out.append(output[b, v - 1, :].view(1, output.shape[-1]))
        grid_vecs = torch.cat(mask_out, dim=0)
        traj_seq, pad_mask, attn_mask = self.repack_grid_vec2traj_seq(grid_vecs, traj_grid_lens)
        return traj_seq, pad_mask, attn_mask

    def repack_grid_vec2traj_seq(self, grid_vecs, traj_grid_lens):
        grid_max_num = torch.max(traj_grid_lens)
        position = 0
        traj_seq = torch.zeros((len(traj_grid_lens), grid_max_num, grid_vecs.shape[-1]), device=cfg.device)
        for i, grid_num in enumerate(traj_grid_lens):
            traj_seq[i, :grid_num] = grid_vecs[position:position + grid_num]
            position += grid_num
        pad_mask = generate_sent_masks(len(traj_grid_lens), grid_max_num, traj_grid_lens, device=cfg.device)
        attn_mask = Transformer.generate_square_subsequent_mask(grid_max_num).to(cfg.device)
        return traj_seq, pad_mask, attn_mask


class GRUTransformerNoClip(GRUTransformerClip):
    def gru_ret2transfseq(self, output, grid_batchs_traj_len, traj_grid_lens):
        grid_max_num = cfg.max_grid_1traj
        transf_in = torch.zeros((output.shape[0], grid_max_num, output.shape[-1]), device=cfg.device)
        for i in range(len(grid_batchs_traj_len)):
            traj_gru_o = output[i]
            traj_node_idx = grid_batchs_traj_len[i, :traj_grid_lens[i]]
            transf_in[i, :traj_grid_lens[i]] = traj_gru_o[traj_node_idx]
        pad_mask = generate_sent_masks(len(traj_grid_lens), grid_max_num, traj_grid_lens, device=cfg.device)
        attn_mask = Transformer.generate_square_subsequent_mask(grid_max_num).to(cfg.device)
        return transf_in, pad_mask, attn_mask

def generate_sent_masks(batch_size, max_seq_length, source_lengths, device, train=True):
    """ Generate sentence masks for encoder hidden states.
        returns enc_masks (Tensor): Tensor of sentence masks of shape (b, max_seq_length),where max_seq_length = max source length """
    if train:
        enc_masks = torch.ones(batch_size, max_seq_length, dtype=torch.float, device=device)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, :src_len] = 0
    else:
        enc_masks = torch.ones(batch_size, max_seq_length, dtype=torch.float, device=device)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, :src_len] = 0
    return enc_masks > 0

class ModelHandler(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.base_model = GRUTransformerNoClip(*args, **kwargs)
        self.loss = None
        self.emb_traj = nn.Parameter(torch.ones((cfg.max_grid_1traj, cfg.dim * 2)),
                                     requires_grad=True)
        self.emb_sum_head = nn.Parameter(torch.ones((cfg.dim * 2)),
                                         requires_grad=True)
        self.t_emb_sum_head = nn.Parameter(torch.zeros((cfg.dim * 2)),
                                           requires_grad=True)
        self.head = cfg.head2
        self.transformer1 = TransformerEncoderNoPos(cfg.dim * 2, num_layer=cfg.layer_M, head=self.head).to(cfg.device)
        self.transformer1_tea = TransformerEncoderNoPos(cfg.dim * 2, head=self.head, num_layer=cfg.layer_G).to(cfg.device)
        self.transformer2 = TransformerEncoderNoPos(cfg.dim * 2, head=1,num_layer=cfg.layer_M).to(cfg.device)
        self.transformer2_tea = TransformerEncoderNoPos(cfg.dim * 2, head=1, num_layer=cfg.layer_G).to(cfg.device)
        stu_attn_mask = torch.diag(torch.ones(cfg.max_grid_1traj))
        stu_attn_mask = torch.cat((torch.cat((torch.zeros_like(stu_attn_mask), stu_attn_mask), dim=1),
                                   torch.cat((stu_attn_mask, torch.zeros_like(stu_attn_mask)), dim=1)), dim=0)
        self.stu_attn_mask = stu_attn_mask.to(torch.int32).to(cfg.device)
        self.tea_np_transfer = MonotonicModule()
        self.stu_np_transfer = MonotonicModule(torch.tensor([-1e8, 0]))

        self.stu = ModelPackage(base=self.base_model, emb_traj=self.emb_traj, emb_head=self.emb_sum_head,
                                t1=self.transformer1, t2=self.transformer2, np_trans=self.stu_np_transfer)
        self.tea = ModelPackage(t1=self.transformer1_tea, t2=self.transformer2_tea, np_trans=self.tea_np_transfer,
                                emb_head=self.t_emb_sum_head)

        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.xavier_uniform_(p.unsqueeze(0))

    def forward(self, data, train=True, batch_size=cfg.batch_size):
        grid_batchs_id, grid_batch_tensor, grid_batchs_traj_len, traj_grid_lens, np_metrixs = data
        trajs_node_emb = self.base_model(grid_batch_tensor, grid_batchs_traj_len, traj_grid_lens)
        if train:
            trajs_node_emb = trajs_node_emb.view(batch_size, 1 + 2 * cfg.sampling_num,
                                                 cfg.max_grid_1traj, cfg.dim * 2)
            all_pad_mask = generate_sent_masks(len(traj_grid_lens), cfg.max_grid_1traj, traj_grid_lens,
                                               device=cfg.device).view(batch_size, 1 + 2 * cfg.sampling_num,
                                                                       cfg.max_grid_1traj)
            base_trajs = trajs_node_emb[:, 0:1]  # 20,1,129,256
            trajs = trajs_node_emb[:, 1:]  # 20,20,129,256
            base_trajs_pad_mask = all_pad_mask[:, 0:1]
            trajs_pad_mask = all_pad_mask[:, 1:]

            cp_base_trajs = base_trajs.repeat(1, cfg.sampling_num * 2, 1, 1)  # 20,20,129,256
            cp_base_pad_mask = base_trajs_pad_mask.repeat(1, cfg.sampling_num * 2, 1)  # 20,20,129
            emb_trajs, emb_pad_mask = self.get_embedding()  # 20,21,129,256

            stu = torch.cat([trajs_node_emb, emb_trajs], dim=2)
            tea = torch.cat([cp_base_trajs, trajs], dim=2)
            tea_attn_mask = np_metrixs

            stu_attn_mask = self.stu_attn_mask.repeat(stu.shape[0] * stu.shape[1], 1, 1)
            # stu_attn_mask = stu_attn_mask.unsqueeze(1).repeat(1, self.head, 1, 1).view(-1,
            #                                                                            static.max_grid_1traj * 2,
            #                                                                            static.max_grid_1traj * 2)
            stu_pad_mask = torch.cat([all_pad_mask, emb_pad_mask], dim=2)

            tea_pad_mask = torch.cat([cp_base_pad_mask, trajs_pad_mask], dim=2)

            # stu, stu_pad_mask, stu_attn_mask = stu.view(20 * 21, static.max_grid_1traj, cfg.dim * 2)
            stu = stu.view(stu.shape[0] * stu.shape[1], cfg.max_grid_1traj * 2, cfg.dim * 2)
            stu_pad_mask = stu_pad_mask.view(stu_pad_mask.shape[0] * stu_pad_mask.shape[1],
                                             2 * cfg.max_grid_1traj)

            tea = tea.view(tea.shape[0] * tea.shape[1], cfg.max_grid_1traj * 2, cfg.dim * 2)
            tea_pad_mask = tea_pad_mask.view(tea_pad_mask.shape[0] * tea_pad_mask.shape[1],
                                             2 * cfg.max_grid_1traj)
            tea_attn_mask = tea_attn_mask.view(tea_attn_mask.shape[0] * tea_attn_mask.shape[1],
                                               cfg.max_grid_1traj * 2, cfg.max_grid_1traj * 2)
            tea_attn_mask = tea_attn_mask.unsqueeze(1).repeat(1, self.head, 1, 1).view(-1,
                                                                                       cfg.max_grid_1traj * 2,
                                                                                       cfg.max_grid_1traj * 2)

            float_stu_attn_mask = self.stu_np_transfer(self.stu_attn_mask)
            float_tea_attn_mask = self.tea_np_transfer(tea_attn_mask)

            stu_np_filtered = self.transformer1(stu, stu_pad_mask, float_stu_attn_mask)
            tea_np_filtered = self.transformer1_tea(tea, tea_pad_mask, float_tea_attn_mask)

            new_stu = self.add_sum_head(stu_np_filtered)
            new_tea = self.add_sum_head_t(tea_np_filtered)

            stu_ret = self.transformer2(new_stu)[:, 0]
            tea_ret = self.transformer2_tea(new_tea)[:, 0]

            return stu_ret.view(cfg.batch_size, -1, cfg.dim * 2), tea_ret.view(cfg.batch_size, -1, cfg.dim * 2)
        else:
            trajs_node_emb = trajs_node_emb.view(cfg.valid_batch_size, cfg.max_grid_1traj, cfg.dim * 2)
            emb_trajs, emb_pad_mask = self.get_embedding(train=False)
            all_pad_mask = generate_sent_masks(len(traj_grid_lens), cfg.max_grid_1traj, traj_grid_lens,
                                               device=cfg.device, train=False)

            stu = torch.cat([trajs_node_emb, emb_trajs], dim=1)
            stu_attn_mask = self.stu_attn_mask.repeat(cfg.valid_batch_size, 1, 1)
            # stu_attn_mask = stu_attn_mask.unsqueeze(1).repeat(1, self.head, 1, 1).view(-1,
            #                                                                            stu_attn_mask.shape[-2],
            #                                                                            stu_attn_mask.shape[-1])

            stu_pad_mask = torch.cat([all_pad_mask, emb_pad_mask], dim=1)
            stu = stu.view(cfg.valid_batch_size, cfg.max_grid_1traj * 2, cfg.dim * 2)
            stu_pad_mask = stu_pad_mask.view(cfg.valid_batch_size, 2 * cfg.max_grid_1traj)
            float_stu_attn_mask = self.stu_np_transfer(self.stu_attn_mask)
            stu_np_filtered = self.transformer1(stu, stu_pad_mask, float_stu_attn_mask)
            new_stu = self.add_sum_head(stu_np_filtered)
            stu_ret = self.transformer2(new_stu)[:, 0]
            return stu_ret

    def get_embedding(self, train=True):
        if train:
            emb = self.emb_traj.unsqueeze(0).unsqueeze(0).repeat(cfg.batch_size, cfg.sampling_num * 2 + 1, 1, 1)
            pad_mask = torch.zeros((cfg.batch_size, cfg.sampling_num * 2 + 1, cfg.max_grid_1traj),
                                   device=cfg.device) > 0
        else:
            emb = self.emb_traj.unsqueeze(0).repeat(cfg.valid_batch_size, 1, 1)
            pad_mask = torch.zeros((cfg.valid_batch_size, cfg.max_grid_1traj), device=cfg.device) > 0

        return emb, pad_mask

    def add_sum_head(self, seq):
        new_seq = torch.zeros((seq.shape[0], seq.shape[1] + 1, seq.shape[2]), device=cfg.device, dtype=torch.float32)
        for i, item in enumerate(seq):
            new_seq[i, 1:len(item) + 1] = item
            new_seq[i, 0] = self.emb_sum_head
        return new_seq

    def add_sum_head_t(self, seq):
        new_seq = torch.zeros((seq.shape[0], seq.shape[1] + 1, seq.shape[2]), device=cfg.device, dtype=torch.float32)
        for i, item in enumerate(seq):
            new_seq[i, 1:len(item) + 1] = item
            new_seq[i, 0] = self.t_emb_sum_head
        return new_seq