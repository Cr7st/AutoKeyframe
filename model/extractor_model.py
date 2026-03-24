import torch.nn as nn
import torch
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, ff_dim, num_heads, num_layers, class_nums,
                 activation='gelu', dropout=0.2, pooling_func = 'mean'):
        super(TransformerEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.activation = activation
        self.input_dim = input_dim
        self.pooling_func = pooling_func
        self.pos_encoding = PositionalEncoding(latent_dim, dropout=dropout)

        trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                        nhead=num_heads,
                                                        dim_feedforward=ff_dim,
                                                        dropout=dropout,
                                                        activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(trans_encoder_layer, num_layers=self.num_layers)
        self.input_linear = nn.Linear(self.input_dim, self.latent_dim)
        self.out_lin1 = nn.Linear(self.latent_dim, 30)
        self.out_lin2 = nn.Linear(30, class_nums)
        self.activate1 = nn.Tanh()

    def forward(self, x, attn_mask=None, padding_mask=None):
        seq_len = x.shape[1]
        if self.pooling_func == 'first':
            x[:, 0] = 0.
        x = self.input_linear(x).permute(1, 0, 2) # [len, bs, n_feats]
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x, mask=attn_mask, src_key_padding_mask=padding_mask)
        x = x.permute((1, 0, 2))  # [bs, len, n_feats]
        x = self.masked_func(x, padding_mask.to(torch.float32), self.pooling_func)
        # x = x[:, 0, :]
        x = self.out_lin1(x)
        x = self.activate1(x)
        out = self.out_lin2(x).view(x.size(0), -1)
        out = nn.functional.log_softmax(out, dim=-1)

        return out

    def extract_features(self, x, attn_mask=None, padding_mask=None):
        seq_len = x.shape[1]
        if self.pooling_func == 'first':
            x[:, 0] = 0.
        x = self.input_linear(x).permute(1, 0, 2)  # [len, bs, n_feats]
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x, mask=attn_mask, src_key_padding_mask=padding_mask)
        x = x.permute((1, 0, 2))  # [bs, len, n_feats]
        x = self.masked_func(x, padding_mask.to(torch.float32), self.pooling_func)
        # x = x[:, 0, :]
        x = self.out_lin1(x)
        x = self.activate1(x)

        return x

    @property
    def trainable_params(self):
        return list(self.parameters())

    @staticmethod
    def masked_func(x, padding_mask, pooling_func = 'mean'):
        mask = 1. - padding_mask
        bs = x.shape[0]
        if pooling_func == 'mean':
            mean_list = [
                x[i, mask[i] == 1].mean(dim=0)[None] for i in range(bs)
            ]
        elif pooling_func == 'max':
            mean_list = [
                x[i, mask[i] == 1].max(dim=0)[0][None] for i in range(bs)
            ]
        elif pooling_func == 'first':
            mean_list = [
                x[i, mask[i] == 1][0][None] for i in range(bs)
            ]
        elif pooling_func == 'last':
            mean_list = [
                x[i, mask[i] == 1][-1][None] for i in range(bs)
            ]
        mean = torch.cat(mean_list, dim=0)
        return mean


class MotionDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer, device, output_size=12, use_noise=None):
        super(MotionDiscriminator, self).__init__()
        self.device = device

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.use_noise = use_noise

        self.recurrent = nn.GRU(input_size, hidden_size, hidden_layer)
        self.linear1 = nn.Linear(hidden_size, 30)
        self.linear2 = nn.Linear(30, output_size)

    def forward(self, motion_sequence, lengths=None, hidden_unit=None):
        # dim (motion_length, num_samples, hidden_size)
        bs, num_frames, nfeats = motion_sequence.shape
        motion_sequence = motion_sequence.permute(1, 0, 2)
        if hidden_unit is None:
            # motion_sequence = motion_sequence.permute(1, 0, 2)
            hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer)
        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)

        # select the last valid, instead of: gru_o[-1, :, :]
        out = gru_o[tuple(torch.stack((lengths-1, torch.arange(bs, device=self.device))))]

        # dim (num_samples, 30)
        lin1 = self.linear1(out)
        # lin1 = torch.tanh(lin1)
        # dim (num_samples, output_size)
        lin2 = self.linear2(lin1)
        lin2 = torch.log_softmax(lin2, dim=-1)
        return lin2

    def initHidden(self, num_samples, layer):
        return torch.randn(layer, num_samples, self.hidden_size, device=self.device, requires_grad=False)

    def extract_features(self, motion_sequence, lengths=None, hidden_unit=None):
        # dim (motion_length, num_samples, hidden_size)
        bs, num_frames, nfeats = motion_sequence.shape
        motion_sequence = motion_sequence.permute(1, 0, 2)
        if hidden_unit is None:
            # motion_sequence = motion_sequence.permute(1, 0, 2)
            hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer)
        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)

        # select the last valid, instead of: gru_o[-1, :, :]
        out = gru_o[tuple(torch.stack((lengths-1, torch.arange(bs, device=self.device))))]

        # dim (num_samples, 30)
        lin1 = self.linear1(out)
        # lin1 = torch.tanh(lin1)
        return lin1