import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import Config


class EmbedLayer(nn.Module):
    def __init__(self, config):
        super(EmbedLayer, self).__init__()
        self.encoder = nn.Embedding(config.n_vocab+1, config.embed_dim)

    def forward(self, x, ):
        return self.encoder(x[0])


class WordCaps(nn.Module):
    def __init__(self, config):
        super(WordCaps, self).__init__()
        self.num_rnn = config.num_rnn
        self.num_units = config.num_units
        # (embedding size, hidden size, rnn_layer)
        self.Lstm = nn.LSTM(config.embed_dim, config.num_units, config.num_rnn,
                            bidirectional=True, batch_first=True, dropout=0)

    def forward(self, x):
        h0 = torch.zeros(self.num_rnn*2, x.size(0),self.num_units).cuda()
        c0 = torch.zeros(self.num_rnn * 2, x.size(0), self.num_units).cuda()
        out, (hn, cn) = self.Lstm(x, (h0, c0))
        # out = F.dropout(out, p=0.8)
        # out = torch.cat([out[:, -1, :self.num_units], out[:, 0, self.num_units:]], dim=1)
        return out


class Capsule(nn.Module):
    def __init__(self, out_casp_num, out_caps_dim, config, iter_num=3):
        super(Capsule, self).__init__()
        self.config = config
        self.out_caps_num = out_casp_num
        self.out_caps_dim = out_caps_dim
        self.iter_num = iter_num
        self.w_rr = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(1, 1, config.num_units, config.intent_dim)), requires_grad=True)
        self.W = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(1, 1024 if out_caps_dim == 512 else 512, self.out_caps_num * self.out_caps_dim,
                                               requires_grad=True)))

    def forward(self, x, caps_ihat=None, re_routing=False):
        # 进行 w * x 的这一步骤作为动态路由算法的输入
        batch_size = x.size(0)
        seq_len = x.size(1)
        caps_uhat = torch.tanh(torch.matmul(x, self.W))
        caps_uhat = caps_uhat.view((batch_size, seq_len, self.out_caps_num, self.out_caps_dim))
        if not re_routing:
            V, S, C, B = masked_routing_iter(caps_uhat, self.iter_num, caps_ihat, w_rr=None)
        else:
            V, S, C, B = masked_routing_iter(caps_uhat, self.iter_num, caps_ihat, w_rr=self.w_rr)
        return V, C, B


def masked_routing_iter(caps_uhat, iter_num, caps_ihat=None, w_rr=None):

    """
    :param caps_uhat: shape(b_sz, seq_len, out_caps_num, out_caps_dim)
    :param iter_num: num of iteration
    :param caps_ihat: using in re-routing as the intent_caps out
    :param w_rr: using in re-routing as a weight matrix
    :return: V_ret shape(b_sz, out_caps_num, out_caps_dim)
    """
    assert iter_num > 0
    batch_size = caps_uhat.size(0)
    seq_len = caps_uhat.size(1)
    out_caps_num = caps_uhat.size(2)
    B = torch.zeros([batch_size, seq_len, out_caps_num]).cuda()
    C_list = list()
    for i in range(iter_num):
        B_logits = B
        C = F.softmax(B, dim=2)
        C = C.unsqueeze(dim=-1)
        weighted_uhat = C * caps_uhat
        C_list.append(C)
        S = torch.sum(weighted_uhat, dim=1)
        V = squash(S, -1)
        V = V.unsqueeze(dim=1)
        if caps_ihat is None:
            B = torch.sum(caps_uhat*V, dim=-1)+B
        else:
            x1 = w_rr.repeat(batch_size, seq_len, 1, 1)
            x2 = torch.matmul(caps_uhat, x1)
            x3 = caps_ihat.repeat(1, seq_len, 1, 1)
            x4 = torch.matmul(x2, x3)
            B = torch.sum(caps_uhat * V, dim=-1) + 0.1 * torch.squeeze(x4, dim=-1) + B
    V_ret = torch.squeeze(V, dim=1)
    S_ret = S
    C_ret = torch.squeeze(torch.stack(C_list), dim=4)
    return V_ret, S_ret, C_ret, B_logits


def squash(in_caps, axes):
    """

    :param in_caps: a tensor
    :param axes: dim along with to apply squash
    :return: vec_squashed: squashed tensor
    """
    s_squared_norm = (in_caps ** 2).sum(axes, keepdim=True)
    scale = torch.sqrt(s_squared_norm+1e-7)
    return in_caps / scale


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.embedding = EmbedLayer(self.config)
        self.wordCaps = WordCaps(self.config)
        self.slotCaps = Capsule(self.config.slot_size, self.config.num_units, self.config,iter_num=self.config.iter_slot)
        self.intentCaps = Capsule(
            self.config.intent_size, self.config.intent_dim, self.config, iter_num=self.config.iter_intent
        )

    def forward(self, x):
        x1 = self.embedding(x)
        x2 = self.wordCaps(x1)
        slot_caps, routing_weight, routing_logits = self.slotCaps(x2, caps_ihat=None)
        slot_p = torch.reshape(routing_logits, [-1, self.config.slot_size])
        intent_caps, intent_routing_weight, _ = self.intentCaps(slot_caps)
        intent = intent_caps
        output = [slot_p, intent, routing_weight, intent_routing_weight]
        if self.config.re_routing:
            pred_intent_index_onehot = F.one_hot(
                torch.argmax(torch.norm(intent_caps, dim=-1), dim=-1), self.config.intent_size
            )
            pred_intent_index_onehot = torch.unsqueeze(pred_intent_index_onehot, 2).repeat(1, 1, intent_caps.size(2))
            # print(pred_intent_index_onehot.device)
            # print(pred_intent_index_onehot.dtype)
            intent_capsule_max = intent_caps.mul(pred_intent_index_onehot.float()).sum(dim=1, keepdim=False)
            # intent_capsule_max = torch.mul(intent_caps, pred_intent_index_onehot).sum(dim=1, keepdim=False)
            caps_ihat = torch.unsqueeze(torch.unsqueeze(intent_capsule_max, dim=1), dim=3)
            slot_caps_new, routing_weight_new, routing_logits_new = self.slotCaps(x2, caps_ihat, re_routing=True)
            slot_p_new = torch.reshape(routing_logits_new, [-1, self.config.slot_size])
            # intent_caps = intent_caps.view(self.config.batch_size, -1)
            # intent = self.fc_intent(intent_caps)
            # intent = torch.norm(intent_caps, dim=-1)
            output = [slot_p_new, intent, routing_weight_new, intent_routing_weight]
        return output



