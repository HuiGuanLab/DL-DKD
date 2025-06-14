import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def safe_div(a, b):
    out = a / b
    out[torch.isnan(out)] = 0
    return out
def onehot(indexes, N=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().long().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    return output


class clip_mse(nn.Module):
    def __init__(self, ):
        super(clip_mse, self).__init__()
    def forward(self, x,target, mask,query_labels):
        loss = torch.pow((x-target),2)
        if len(loss.shape)==3:
            loss = loss.sum(dim=1)
        else:
            loss = loss.sum(dim=-1)
        loss = loss.mean()
        return loss

class clip_mse_pos_pair(nn.Module):
    def __init__(self, ):
        super(clip_mse_pos_pair, self).__init__()
    def forward(self, x,target, mask,query_labels):
        loss = 0
        for idx, label in enumerate(query_labels):
            m = mask[label]
            m = torch.nonzero(m > 0).shape[0]
            p = x[idx, :m, label]
            q = target[idx, :m, label]
            loss += torch.sum(torch.pow((p-q),2),dim=0) / m
        # loss /= x.shape[0]
        return loss

class clip_mse_max_pos_pair(nn.Module):
    def __init__(self, ):
        super(clip_mse_max_pos_pair, self).__init__()
    def forward(self, x,target, mask,query_labels):
        loss = 0
        for idx, label in enumerate(query_labels):
            # m = mask[label]
            # m = torch.nonzero(m > 0).shape[0]
            p = x[idx, label]
            q = target[idx, label]
            loss += torch.pow((p-q),2)
        # loss = loss.mean()
        loss /= x.shape[0]
        return loss

class clip_mse_only_pos_max(nn.Module):
    def __init__(self):
        super(clip_mse_only_pos_max, self).__init__()
    def forward(self, x, target, mask, query_labels, cap_ids=None):
        loss = 0
        for idx, label in enumerate(query_labels):
            m = mask[label]
            m = torch.nonzero(m > 0).shape[0]
            p = x[idx, :m, label]
            q = target[idx, :m, label]
            max_idx = torch.argmax(q)
            p = p[max_idx]
            q = q[max_idx]
            loss += torch.pow((p-q),2)
        return loss

class clip_kl_only_pos(nn.Module):
    def __init__(self):
        super(clip_kl_only_pos, self).__init__()
        self.tempeture = 0.2
        self.mse_loss = nn.MSELoss(reduction= "sum")
        # self.loss = nn.L1Loss(reduction= "mean")
    def forward(self, x,target,mask,query_labels):
        loss=0
        for idx,label in enumerate(query_labels):
            m = mask[label]
            m = torch.nonzero(m > 0).shape[0]
            p = x[idx, :m, label]
            q = target[idx, :m, label]
            #KL loss
            logp_x = F.log_softmax(p / self.tempeture , dim=-1)
            p_y = F.softmax(q /self.tempeture, dim=-1)
            loss += F.kl_div(logp_x, p_y, reduction='sum')

        return loss


class clip_nce_soft(nn.Module):
    def __init__(self, reduction='mean'):
        super(clip_nce_soft, self).__init__()
        self.reduction = reduction

    def forward(self, labels, label_dict, q2ctx_scores=None, sims=None, alpha=None, belta=0.8, contexts=None, queries=None):
        """
        the forward propagation process and calculate the loss value.

        参数:
        - label_dict: Mapping text and video
        - q2ctx_scores: Similarity score
        - sims: Similarity score for soft operation
        - alpha: nce loss self-distillation, data partition threshold
        - belta: nce loss self-distillation, GroundTruth and soft weighted and weighted
        返回:
        - loss: loss value
        """

        # Get batch size of query and video
        query_bsz = q2ctx_scores.shape[0]
        vid_bsz = q2ctx_scores.shape[1]

        # Calculate number of queries and videos in hard and soft parts
        hardQ = math.floor(alpha * query_bsz)
        softQ = query_bsz - hardQ

        hardV = math.floor(alpha * vid_bsz)
        softV = vid_bsz - hardV

        # Initialize the GroundTruth matrix
        I_ij = torch.zeros(query_bsz, vid_bsz).to(q2ctx_scores.device)

        # Update the GroundTruth matrix based on label dictionary
        for i, label in label_dict.items():
            I_ij[label, i] = 1

        # Compute soft targets for text
        I_ij_Q = I_ij.clone()
        sims_t = torch.softmax(sims, dim=-1)
        I_ij_Q[hardQ:, :] = torch.clamp((1 - belta) * sims_t[hardQ:, :] + belta * I_ij_Q[hardQ:, :], min=0)

        # Compute soft targets for video
        I_ij_V = I_ij.T.clone()
        sims_v = torch.softmax(sims.T, dim=-1)
        I_ij_V[hardV:, :] = torch.clamp((1 - belta) * sims_v[hardV:, :] + belta * I_ij_V[hardV:, :], min=0)

        # Calculate loss for the hard part of t2v
        exp_q2ctx_scores_hard_t2v = torch.exp(q2ctx_scores[:hardQ, :])
        t2v_nominator_hard_ = (I_ij_Q[:hardQ, :] * torch.log(exp_q2ctx_scores_hard_t2v)).sum()
        t2v_nominator_hard = (I_ij_Q[:hardQ, :] * q2ctx_scores[:hardQ, :]).sum()
        t2v_denominator_hard = (I_ij_Q[:hardQ, :] * torch.logsumexp(q2ctx_scores[:hardQ, :], dim=1, keepdim=True)).sum()

        # Calculate loss for the soft part of t2v
        exp_q2ctx_scores_soft_t2v = torch.exp(q2ctx_scores[hardQ:, :])
        t2v_nominator_soft_ = (I_ij_Q[hardQ:, :] * torch.log(exp_q2ctx_scores_soft_t2v)).sum()
        t2v_nominator_soft = (I_ij_Q[hardQ:, :] * q2ctx_scores[hardQ:, :]).sum()
        t2v_denominator_soft = (I_ij_Q[hardQ:, :] * torch.logsumexp(q2ctx_scores[hardQ:, :], dim=1, keepdim=True)).sum()

        # Initialize numerator and denominator for the hard part of v2t
        v2t_nominator_hard = torch.zeros(1).to(q2ctx_scores.device)
        v2t_denominator_hard = torch.zeros(1).to(q2ctx_scores.device)
        # Calculate loss for the hard part of v2t
        for i, label in label_dict.items():
            if i < hardV:
                v2t_nominator_hard += torch.logsumexp(torch.log(I_ij_V[i, :] + 1e-12) + q2ctx_scores[:, i], dim=0)
                v2t_denominator_hard += torch.logsumexp(q2ctx_scores[:, i], dim=0)

        v2t_nominator_soft = torch.zeros(1).to(q2ctx_scores.device)
        v2t_denominator_soft = torch.zeros(1).to(q2ctx_scores.device)
        # Calculate loss for the soft part of v2t
        for i, label in label_dict.items():
            if i >= hardV:
                v2t_nominator_soft += torch.logsumexp(torch.log(I_ij_V[i, :] + 1e-12) + q2ctx_scores[:, i], dim=0)
                v2t_denominator_soft += torch.logsumexp(q2ctx_scores[:, i], dim=0)

        # Calculate final loss according to reduction parameter
        if self.reduction == 'mean':
            soft_loss = 0.
            hard_loss = 0.
            # Calculate loss for hard part
            if hardQ != 0 and hardV != 0:
                hard_loss_t2v = (t2v_denominator_hard - t2v_nominator_hard) / hardQ
                hard_loss_v2t = (v2t_denominator_hard - v2t_nominator_hard) / hardV
                hard_loss = hard_loss_t2v + hard_loss_v2t

            # Calculate loss for soft part
            if softQ != 0 and softV != 0:
                soft_loss_t2v = (t2v_denominator_soft - t2v_nominator_soft) / softQ
                soft_loss_v2t = (v2t_denominator_soft - v2t_nominator_soft) / softV
                soft_loss = soft_loss_t2v + soft_loss_v2t

            # Combine hard and soft losses
            loss = alpha * hard_loss + (1 - alpha) * soft_loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("NaN detected in value:", loss)

        else:
            # Directly calculate loss if mean reduction is not used
            hard_loss = (t2v_denominator_hard - t2v_nominator_hard) + (v2t_denominator_hard - v2t_nominator_hard)
            soft_loss = (t2v_denominator_soft - t2v_nominator_soft) + (v2t_denominator_soft - v2t_nominator_soft)
            loss = alpha * hard_loss + (1 - alpha) * soft_loss

        return loss

class clip_nce(nn.Module):
    def __init__(self, reduction='mean'):
        super(clip_nce, self).__init__()
        self.reduction = reduction

    def forward(self,labels, label_dict, q2ctx_scores=None, contexts=None, queries=None):

        query_bsz = q2ctx_scores.shape[0]
        vid_bsz = q2ctx_scores.shape[1]
        diagnoal = torch.arange(query_bsz).to(q2ctx_scores.device)
        t2v_nominator = q2ctx_scores[diagnoal, labels]

        t2v_nominator = torch.logsumexp(t2v_nominator.unsqueeze(1), dim=1)
        t2v_denominator = torch.logsumexp(q2ctx_scores, dim=1)

        v2t_nominator = torch.zeros(vid_bsz).to(q2ctx_scores)
        v2t_denominator = torch.zeros(vid_bsz).to(q2ctx_scores)

        for i, label in label_dict.items():
            v2t_nominator[i] = torch.logsumexp(q2ctx_scores[label, i], dim=0)

            v2t_denominator[i] = torch.logsumexp(q2ctx_scores[:, i], dim=0)
        if self.reduction:
            return torch.mean(t2v_denominator - t2v_nominator)+torch.mean(v2t_denominator - v2t_nominator)
        else:
            return denominator - nominator

class frame_nce(nn.Module):
    def __init__(self, reduction='mean'):
        super(frame_nce, self).__init__()
        self.reduction = reduction

    def forward(self, q2ctx_scores=None, contexts=None, queries=None):
        if q2ctx_scores is None:
            assert contexts is not None and queries is not None
            x = torch.matmul(contexts, queries.t())
            device = contexts.device
            bsz = contexts.shape[0]
        else:
            x = q2ctx_scores
            device = q2ctx_scores.device
            bsz = q2ctx_scores.shape[0]

        x = x.view(bsz, bsz, -1)
        nominator = x * torch.eye(x.shape[0], dtype=torch.float32, device=device)[:, :, None]
        nominator = nominator.sum(dim=1)

        nominator = torch.logsumexp(nominator, dim=1)

        denominator = torch.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)
        if self.reduction:
            return torch.mean(denominator - nominator)
        else:
            return denominator - nominator



class TrainablePositionalEncoding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, max_position_embeddings, hidden_size, dropout=0.1):
        super(TrainablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(input_feat + position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def add_position_emb(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        return input_feat + position_embeddings


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""
    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [nn.Dropout(dropout), nn.Linear(in_hsz, out_hsz)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


class BertLayer(nn.Module):
    def __init__(self, config, use_self_attention=True):
        super(BertLayer, self).__init__()
        self.use_self_attention = use_self_attention
        if use_self_attention:
            self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        """
        Args:
            hidden_states:  (N, L, D)
            attention_mask:  (N, L) with 1 indicate valid, 0 indicates invalid
        """
        if self.use_self_attention:
            attention_output = self.attention(hidden_states, attention_mask)
        else:
            attention_output = hidden_states
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask=None):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, L)
        """
        self_output = self.self(input_tensor, input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Sequential(nn.Linear(config.hidden_size, config.intermediate_size), nn.ReLU(True))

    def forward(self, hidden_states):
        return self.dense(hidden_states)


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (
                config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states, key_states, value_states, attention_mask=None):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)
        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        # attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)
        # transpose
        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
            attention_scores = attention_scores + attention_mask
        # attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # compute output context
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



