import copy
from typing import Optional
import torch.nn as nn
import torch
import numpy as np
from torch import Tensor


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


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)

class HintBlock(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.ModuleList([
            nn.Linear(self.input_feats, 64),
            nn.Linear(64, 128),
            nn.Linear(128, 256),
            nn.Tanh(),
            zero_module(nn.Linear(256, self.latent_dim))
        ])

    def forward(self, x):
        x = x.permute((1, 0, 2))

        for module in self.poseEmbedding:
            x = module(x)  # [seqlen, bs, d]
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, ff_dim, num_heads, num_layers, condition_dims, 
                 conditions = None, activation='gelu', dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.activation = activation
        self.input_dim = input_dim
        self.conditions = conditions
        self.pos_encoding = PositionalEncoding(latent_dim, dropout=dropout)
        self.timestep_embedder = TimestepEmbedder(latent_dim, self.pos_encoding)

        trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                        nhead=num_heads,
                                                        dim_feedforward=ff_dim,
                                                        dropout=dropout,
                                                        activation=activation)

        self.transformer_encoder = ControllableTransformerEncoder(trans_encoder_layer, num_layers=self.num_layers)
        self.condition_embedders = nn.ModuleList()
        for cond_dim in condition_dims:
            self.condition_embedders.append(nn.Linear(cond_dim, latent_dim))
        self.input_linear = nn.Linear(self.input_dim, self.latent_dim)
        self.output_linear = nn.Linear(self.latent_dim, self.input_dim)

    @property
    def trainable_params(self):
        return list(self.parameters())

    def forward(self, x, conditions, timestep, attn_mask=None, padding_mask=None, **kwargs):
        time_emb = self.timestep_embedder(timestep)
        cond_emb_list = []
        seq_len = x.shape[1]
        x = self.input_linear(x).permute(1, 0, 2) # [len, bs, n_feats]
        bias = 0
        for i, (key, cond) in enumerate(conditions.items()):
            if len(cond.shape) == 2:
                cond = cond.unsqueeze(1)
            if self.conditions is not None and key not in self.conditions:
                bias -= 1
                continue
            cond_emb_list.append(self.condition_embedders[i+bias](cond).permute(1, 0, 2))
        cond_emb = torch.cat(cond_emb_list, dim=0)

        x = torch.cat([cond_emb, time_emb, x], dim=0)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x, mask=attn_mask, src_key_padding_mask=padding_mask)
        x = self.output_linear(x)
        x = x.permute((1, 0, 2))  # [bs, len, n_feats]

        return x[:, -seq_len:, :]


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class ControlNet(TransformerEncoder):
    def __init__(self, input_dim, latent_dim, ff_dim, num_heads, num_layers, condition_dims,
                 activation='gelu', dropout=0.1, hint_dim=3):
        super(ControlNet, self).__init__(input_dim, latent_dim, ff_dim, num_heads, num_layers, condition_dims,
                 activation, dropout)
        self.hint_dim = hint_dim
        self.control_transformer_encoder = copy.deepcopy(self.transformer_encoder)
        self.control_transformer_encoder.return_intermediate = True

        self.input_zero_conv = HintBlock('input', 3, self.latent_dim)
        self.zero_convs = nn.ModuleList([zero_module(nn.Linear(self.latent_dim, self.latent_dim)) for _ in range(self.num_layers)])

    @property
    def trainable_params(self):
        return (list(self.control_transformer_encoder.parameters()) + list(self.input_zero_conv.parameters()) +
                [p for conv in self.zero_convs for p in conv.parameters()])

    def freeze_original_params(self):
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False
        for param in self.pos_encoding.parameters():
            param.requires_grad = False
        for param in self.timestep_embedder.parameters():
            param.requires_grad = False
        for param in self.condition_embedders.parameters():
            param.requires_grad = False
        for param in self.input_linear.parameters():
            param.requires_grad = False
        for param in self.output_linear.parameters():
            param.requires_grad = False

    def forward(self, x, conditions, timestep, attn_mask=None, padding_mask=None, hint=None, hint_mask=None):
        time_emb = self.timestep_embedder(timestep)
        cond_emb_list = []
        seq_len = x.shape[1]
        x = self.input_linear(x).permute(1, 0, 2)
        for i, cond in enumerate(conditions.values()):
            if len(cond.shape) == 2:
                cond = cond.unsqueeze(1)
            cond_emb_list.append(self.condition_embedders[i](cond).permute(1, 0, 2))
        cond_emb = torch.cat(cond_emb_list, dim=0)
        original_x = torch.cat([cond_emb, time_emb, x], dim=0)
        original_x = self.pos_encoding(original_x)

        if hint_mask is not None:
            hint = self.input_zero_conv(hint)
            hint = hint * hint_mask[..., 0].permute(1, 0).unsqueeze(-1)
            control_x = x + hint
            control_x = torch.cat([cond_emb, time_emb, control_x], dim=0)
            control_x = self.pos_encoding(control_x)
            out = self.control_transformer_encoder(control_x, mask=attn_mask, src_key_padding_mask=padding_mask)
            control = []
            for i, zero_conv in enumerate(self.zero_convs):
                control.append(zero_conv(out[i]) * 10)
            control = torch.stack(control)
        else:
            control = None
        x = self.transformer_encoder(original_x, mask=attn_mask, src_key_padding_mask=padding_mask, control=control)
        x = self.output_linear(x)
        x = x.permute((1, 0, 2))  # [bs, len, n_feats]

        return x[:, -seq_len:, :]



class ControllableTransformerEncoder(nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=False, return_intermediate=False):
        super().__init__(encoder_layer, num_layers, norm, enable_nested_tensor)
        self.return_intermediate = return_intermediate

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, control: Optional[Tensor] = None) -> Tensor:
        output = src
        intermediate = []
        convert_to_nested = False
        first_layer = self.layers[0]
        if isinstance(first_layer, torch.nn.TransformerEncoderLayer):
            if (not first_layer.norm_first and not first_layer.training and
                    first_layer.self_attn.batch_first and
                    first_layer.self_attn._qkv_same_embed_dim and first_layer.activation_relu_or_gelu and
                    first_layer.norm1.eps == first_layer.norm2.eps and
                    src.dim() == 3 and self.enable_nested_tensor):
                if src_key_padding_mask is not None and not output.is_nested and mask is None:
                    tensor_args = (
                        src,
                        first_layer.self_attn.in_proj_weight,
                        first_layer.self_attn.in_proj_bias,
                        first_layer.self_attn.out_proj.weight,
                        first_layer.self_attn.out_proj.bias,
                        first_layer.norm1.weight,
                        first_layer.norm1.bias,
                        first_layer.norm2.weight,
                        first_layer.norm2.bias,
                        first_layer.linear1.weight,
                        first_layer.linear1.bias,
                        first_layer.linear2.weight,
                        first_layer.linear2.bias,
                    )
                    if not torch.overrides.has_torch_function(tensor_args):
                        if not torch.is_grad_enabled() or all([not x.requires_grad for x in tensor_args]):
                            if output.is_cuda or 'cpu' in str(output.device):
                                convert_to_nested = True
                                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not())

        for i, mod in enumerate(self.layers):
            if convert_to_nested:
                output = mod(output, src_mask=mask)
            else:
                output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

            if control is not None:
                output = output + control[i]
            if self.return_intermediate:
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)

        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        if self.norm is not None:
            output = self.norm(output)

        return output
