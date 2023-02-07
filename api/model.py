from torch.nn import functional
import numpy as np
import torch
from torch import nn
from torch import nn
from torch.nn import init
from torch.nn.utils import rnn as rnn_utils
import math

import abc


class LinearModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        layer = []
        n=0
        temp_output_size = input_size//16
        while True:
            if temp_output_size <= output_size:
                layer.append(nn.Linear(input_size, output_size))
                break
            else:
                layer.append(nn.Linear(input_size, temp_output_size))
                # if n < 1:
                # layer.append(nn.Tanh())
                    # layer.append(SELayer(channel=input_size))
                    # n += 1
                # layer.append(nn.Linear(input_size, temp_output_size))
                input_size = temp_output_size
                temp_output_size = temp_output_size//16

        self.fcn_layer = nn.ModuleList(layer)

    def forward(self, x):
        # print("22 >>>>>>>>", x.shape)
        for layer in self.fcn_layer:
            x = layer(x)
        return x


class BiLSTM(nn.Module):

    def __init__(self, embedding_size=768, hidden_dim=512, rnn_layers=1, dropout=0.5, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_size,
            hidden_dim // 2,
            rnn_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, input_,):
        output, (hidden, _) = self.lstm(input_)
        return output, hidden


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x len_q x d_k]
        # k: [b_size x len_k x d_k]
        # v: [b_size x len_v x d_v] note: (len_k == len_v)
        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale_factor  # attn: [b_size x len_q x len_k]
        if attn_mask is not None:
            print(attn_mask.size(), attn.size())
            assert attn_mask.size() == attn.size()
            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        outputs = torch.bmm(attn, v) # outputs: [b_size x len_q x d_v]

        return outputs, attn


class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(d_hid), requires_grad=True)
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True,)
        std = z.std(dim=-1, keepdim=True,)
        ln_out = (z - mean.expand_as(z)) / (std.expand_as(z) + self.eps)
        ln_out = self.gamma.expand_as(ln_out) * ln_out + self.beta.expand_as(ln_out)

        return ln_out


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(_MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads
        self.w_q = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_k))
        self.w_k = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_k))
        self.w_v = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_v))

        self.attention = ScaledDotProductAttention(d_k, dropout)

        init.xavier_normal(self.w_q)
        init.xavier_normal(self.w_k)
        init.xavier_normal(self.w_v)

    def forward(self, q, k, v, attn_mask=None):
        (d_k, d_v, d_model, n_heads) = (self.d_k, self.d_v, self.d_model, self.n_heads)
        b_size = k.size(0)

        q_s = q.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_q x d_model]
        k_s = k.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_k x d_model]
        v_s = v.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_v x d_model]

        q_s = torch.bmm(q_s, self.w_q).view(b_size * n_heads, -1, d_k)  # [b_size * n_heads x len_q x d_k]
        k_s = torch.bmm(k_s, self.w_k).view(b_size * n_heads, -1, d_k)  # [b_size * n_heads x len_k x d_k]
        v_s = torch.bmm(v_s, self.w_v).view(b_size * n_heads, -1, d_v)  # [b_size * n_heads x len_v x d_v]

        # perform attention, result_size = [b_size * n_heads x len_q x d_v]
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_heads, 1, 1)
        outputs, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask)

        # return a list of tensors of shape [b_size x len_q x d_v] (length: n_heads)
        return torch.split(outputs, b_size, dim=0), attn


class Linear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True):
        super(Linear, self).__init__(in_features, out_features, bias=bias)
        init.orthogonal_(self.weight)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.attention = _MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.proj = Linear(n_heads * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, q, k, v, attn_mask):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)
        residual = q
        # outputs: a list of tensors of shape [b_size x len_q x d_v] (length: n_heads)
        outputs, attn = self.attention(q, k, v, attn_mask=attn_mask)
        # concatenate 'n_heads' multi-head attentions
        outputs = torch.cat(outputs, dim=-1)
        # project back to residual size, result_size = [b_size x len_q x d_model]
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)

        return self.layer_norm(residual + outputs), attn


class LMRBiLSTMAttnCRF(nn.Module):

    def __init__(self, embedding_size, hidden_dim, rnn_layers,lstm_dropout, device, key_dim=64, val_dim=64, num_output=64, num_heads=3, attn_dropout=0.3):
        super(LMRBiLSTMAttnCRF, self).__init__()
        self.encoder = BiLSTM(embedding_size=embedding_size, hidden_dim=hidden_dim,
                              rnn_layers=rnn_layers, dropout=lstm_dropout)
        embedding_size = hidden_dim

        self.decoder = BiLSTM(embedding_size=embedding_size, hidden_dim=hidden_dim,
                              rnn_layers=rnn_layers, dropout=lstm_dropout, bidirectional=False)
        hidden_dim = hidden_dim//2
        self.attn = MultiHeadAttention(key_dim, val_dim, hidden_dim, num_heads, attn_dropout)
        # self.linear = ResidualBlock(input_size=(hidden_dim*num_output), output_size=num_output)
        self.linear = LinearModel(input_size=(hidden_dim*num_output), output_size=num_output)
        # self.activation = nn.Sigmoid()
        self.to(device)

    def forward(self, batch, z):
        # print(f"{0} >>>>> {batch}")
        output, _ = self.encoder.forward(batch)
        # print(f"{1} >>>>> {output}")
        output, _ = self.decoder.forward(output)
        # print(f"{2} >>>>> {output}")
        output, _ = self.attn(output, output, output, attn_mask=None)
        # print(f"{3} >>>>> {output}")
        b, r, c = output.shape
        output = torch.reshape(output, (b, c * r))
        output = self.linear(output)
        # output = self.activation(output)
        return output


class LMRBiLSTMAttnCRF3(nn.Module):

    def __init__(self, embedding_size, hidden_dim, rnn_layers,lstm_dropout, device,
                 key_dim=64, val_dim=64, num_output=64, num_heads=3, attn_dropout=0.3):
        super(LMRBiLSTMAttnCRF, self).__init__()
        self.lstm = BiLSTM(embedding_size=embedding_size, hidden_dim=hidden_dim, rnn_layers=rnn_layers, dropout=lstm_dropout)
        # self.pos_lstm = BiLSTM(embedding_size=num_output, hidden_dim=num_output, rnn_layers=rnn_layers, dropout=lstm_dropout)
        self.attn = MultiHeadAttention(key_dim, val_dim, hidden_dim, num_heads, attn_dropout)
        # self.linear = ResidualBlock(input_size=(hidden_dim*num_output), output_size=num_output)
        self.linear = LinearModel(input_size=(hidden_dim*num_output), output_size=num_output)
        # self.activation = nn.Sigmoid()
        self.to(device)

    def forward(self, batch, z):
        # z = torch.round(z + 0.6)
        # b1, c1 = z.shape
        # z = torch.reshape(z, (b1, 1, c1))
        output, _ = self.lstm.forward(batch)
        output, _ = self.attn(output, output, output, attn_mask=None)
        #
        # output1, _ = self.pos_lstm(z)
        b, r, c = output.shape
        # b1, r1, c1 = output1.shape
        output = torch.reshape(output, (b, c * r))
        # output1 = torch.reshape(output1, (b1, c1 * r1))
        # output = self.activation(self.linear(output)+ output1)
        # b, r, c = batch.shape
        # output = torch.reshape(batch, (b, c * r))
        output = self.linear(output)
        # output = self.activation(output)
        return output


class LMRBiLSTMAttnCRF2(nn.Module):

    def __init__(self, embedding_size, hidden_dim, rnn_layers,lstm_dropout, device, key_dim=64, val_dim=64, num_output=64, num_heads=3, attn_dropout=0.3):
        super(LMRBiLSTMAttnCRF2, self).__init__()
        # self.lstm = BiLSTM(embedding_size=embedding_size, hidden_dim=hidden_dim, rnn_layers=rnn_layers, dropout=lstm_dropout)
        # self.pos_lstm = BiLSTM(embedding_size=num_output, hidden_dim=num_output, rnn_layers=rnn_layers, dropout=lstm_dropout)
        self.attn = MultiHeadAttention(key_dim, val_dim, embedding_size, num_heads, attn_dropout)
        # self.linear = ResidualBlock(input_size=(hidden_dim*num_output), output_size=num_output)
        self.linear = LinearModel(input_size=(hidden_dim*num_output), output_size=num_output)
        # self.activation = nn.Sigmoid()
        self.to(device)

    def forward(self, batch, z):
        # z = torch.round(z + 0.6)
        b, r, c = batch.shape
        # z = torch.reshape(z, (b1, 1, c1))
        # output, _ = self.lstm.forward(batch)
        output = torch.reshape(batch, (b, r*c))
        output, _ = self.attn(output, output, output, attn_mask=None)
        #
        # output1, _ = self.pos_lstm(z)
        b, r, c = output.shape
        # b1, r1, c1 = output1.shape
        output = torch.reshape(output, (b, c * r))
        # output1 = torch.reshape(output1, (b1, c1 * r1))
        # output = self.activation(self.linear(output)+ output1)
        # b, r, c = batch.shape
        # output = torch.reshape(batch, (b, c * r))
        output = self.linear(output)
        # output = self.activation(output)
        return output

