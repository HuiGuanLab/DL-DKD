import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
from method.model_components import BertAttention, LinearLayer, BertSelfAttention, TrainablePositionalEncoding
from method.model_components import clip_nce, frame_nce, clip_mse, clip_kl, clip_info, clip_mse_max_pos_pair,clip_kl_only_pos, \
                                    clip_mse_only_pos_max, \
                                    clip_dis_feat,clip_mse_pos_pair, clip_score_matrix , \
                                    clip_scl_modified



class MS_SL_Net(nn.Module):
    def __init__(self, config, opt):
        super(MS_SL_Net, self).__init__()
        self.config = config
        self.epoch = 0
        self.use_clip_guiyi = opt.use_clip_guiyi
        self.use_clip = opt.use_clip
        self.double_branch = opt.double_branch

        if self.double_branch:
            self.A_query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l,
                                                               hidden_size=config.A_hidden_size,dropout=config.input_drop)
            self.A_frame_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                               hidden_size=config.A_hidden_size,dropout=config.input_drop)
            self.A_query_input_proj = LinearLayer(config.query_input_size, config.A_hidden_size, layer_norm=True,
                                                dropout=config.input_drop, relu=True)
            self.A_query_encoder = BertAttention(
                edict(hidden_size=config.A_hidden_size, intermediate_size=config.A_hidden_size,
                      hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                      attention_probs_dropout_prob=config.drop))
            self.A_frame_encoder = BertAttention(
                edict(hidden_size=config.A_hidden_size, intermediate_size=config.A_hidden_size,
                      hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                      attention_probs_dropout_prob=config.drop))

            self.A_frame_input_proj = LinearLayer(config.visual_input_size, config.A_hidden_size, layer_norm=True,
                                                dropout=config.input_drop, relu=True)
            self.A_modular_vector_mapping = nn.Linear(config.A_hidden_size, out_features=1, bias=False)

            self.A_mapping_linear = nn.Linear(config.A_hidden_size, out_features=config.A_hidden_size)


        self.B_query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l,
                                                             hidden_size=config.B_hidden_size,
                                                             dropout=config.input_drop)
        self.B_modular_vector_mapping = nn.Linear(config.B_hidden_size, out_features=1, bias=False)
        self.B_query_encoder = BertAttention(
            edict(hidden_size=config.B_hidden_size, intermediate_size=config.B_hidden_size,
                  hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                  attention_probs_dropout_prob=config.drop))

        self.B_query_input_proj = LinearLayer(config.query_input_size, config.B_hidden_size, layer_norm=True,
                                              dropout=config.input_drop, relu=True)

        self.B_frame_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                           hidden_size=config.B_hidden_size, dropout=config.input_drop)
        self.B_frame_input_proj = LinearLayer(config.visual_input_size, config.B_hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.B_frame_encoder = BertAttention(edict(hidden_size=config.B_hidden_size, intermediate_size=config.B_hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop))

        self.B_mapping_linear = nn.Linear(config.B_hidden_size, out_features=config.B_hidden_size)



        self.nce_criterion = clip_nce(reduction='mean')

        self.reset_parameters()
        self.use_clip = opt.use_clip

        # self.clip_loss = clip_mse()
        # self.clip_loss = clip_mse_pos_pair()

        # self.clip_loss = clip_kl()
        self.clip_loss = clip_kl_only_pos()

        # self.clip_loss = clip_info()
        self.clip_dis_feat_loss = clip_dis_feat()

        # self.clip_feat_loss = clip_scl()
        # self.clip_feat_loss = clip_scl_modified()

        # self.clip_matrix_loss = clip_score_matrix()

        self.scale_weight = opt.loss_scale_weight
        if opt.decay_way >3 and opt.decay_way <7:
            self.init_weight = opt.loss_init_weight
        else:
            self.init_weight = 0

        self.weight=1

    def reset_parameters(self):
        """ Initialize the weights."""
        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """use_hard_negative: bool; hard_pool_size: int, """
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size

    def forward(self, frame_video_feat, clip_video_features, frame_video_mask, query_feat,
                query_mask, clip_query_feat, query_labels, cap_ids):

        encoded_frame_feat_1, encoded_frame_feat_2 = self.encode_context(frame_video_feat, frame_video_mask)
        video_query, video_query_B = self.encode_query(query_feat, query_mask)
        max_video_score, max_video_score_, max_predict_video_score, max_predict_video_score_ \
            = self.get_pred_from_raw_query(
            video_query, video_query_B, encoded_frame_feat_1, encoded_frame_feat_2, frame_video_mask,
            return_query_feats=True)
        video_query_B = video_query_B.squeeze(1)

        label_dict = {}
        for index, label in enumerate(query_labels):
            if label in label_dict:
                label_dict[label].append(index)
            else:
                label_dict[label] = []
                label_dict[label].append(index)

        frame_nce_loss = 0
        frame_trip_loss = 0
        if self.double_branch:
            frame_nce_loss = 0.04 * self.nce_criterion(query_labels, label_dict, max_video_score_)
            frame_trip_loss = self.get_clip_triplet_loss(max_video_score, query_labels)

        c_guide_loss=0
        c_scl_loss=0
        c_feat_dis_loss=0
        clip_feat_loss=0
        c_matrix_loss=0
        c_tri_loss = self.get_clip_triplet_loss(max_predict_video_score, query_labels)

        if self.use_clip:
            _, clip_score = self.get_clip_scale_scores(clip_query_feat,clip_video_features,frame_video_mask)

            # c_guide_loss = self.scale_weight * self.weight * self.clip_dis_feat_loss(F.normalize(encoded_frame_feat_2, dim=-1),
            #                                              F.normalize(clip_video_features, dim=-1), frame_video_mask)
            # c_guide_loss += self.scale_weight * self.weight * self.clip_dis_feat_loss(F.normalize(video_query_B.squeeze(1), dim=-1),
            #                                               F.normalize(clip_query_feat, dim=-1))

            #KL loss,敏松原代码
            c_guide_loss = (self.scale_weight * self.weight + self.init_weight) * self.clip_loss(max_predict_video_score_,clip_score, frame_video_mask,query_labels) #分布进行对齐

            # c_guide_loss = 0.2 * self.clip_feat_loss(encoded_frame_feat_2,clip_video_features, frame_video_mask) #和clip进行对齐
            # c_guide_loss = 0.1 * self.clip_feat_loss(encoded_frame_feat_2,clip_video_features, frame_video_mask)
            # c_scl_loss = 1 * self.clip_feat_loss(encoded_frame_feat_2,clip_video_features, frame_video_mask)

            # c_matrix_loss = self.weight * self.clip_matrix_loss(encoded_frame_feat_2,clip_video_features,frame_video_mask)

        max_predict_video_score_ = self.get_unnormalized_clip_scale_scores(video_query_B, encoded_frame_feat_2, frame_video_mask)
        c_nce_loss = 0.04 * self.nce_criterion(query_labels, label_dict, max_predict_video_score_)

        # loss = frame_nce_loss + frame_trip_loss + c_tri_loss + c_nce_loss + 2 * c_guide_loss + 0.02 * c_scl_loss + 0.004 * c_feat_dis_loss

        loss = c_tri_loss + c_guide_loss + c_scl_loss + c_matrix_loss + c_nce_loss
        if self.double_branch:
            loss += frame_nce_loss + frame_trip_loss

        return loss, {"loss_overall": float(loss), 'clip_trip_loss': c_tri_loss,
                      'frame_nce_loss': frame_nce_loss, 'clip_guide_loss': c_guide_loss,
                      'frame_trip_loss':frame_trip_loss ,'clip_feat_loss':clip_feat_loss,
                      'clip_feat_dis_loss':c_feat_dis_loss,'c_matrix_loss':c_matrix_loss,
                      'clip_scl_loss': c_scl_loss,'c_nce_loss':c_nce_loss
                      }



    def encode_query(self, query_feat, query_mask):
        B_encoded_query = self.encode_input(query_feat, query_mask, self.B_query_input_proj, self.B_query_encoder,
                                            self.B_query_pos_embed)  # (N, Lq, D)

        B_video_query = self.get_modularized_queries(B_encoded_query, query_mask, t=False)  # (N, D) * 1
        if self.double_branch:
            A_encoded_query = self.encode_input(query_feat, query_mask, self.B_query_input_proj, self.A_query_encoder,
                                              self.A_query_pos_embed)  # (N, Lq, D)
            A_video_query = self.get_modularized_queries(A_encoded_query, query_mask)  # (N, D) * 1
            return A_video_query, B_video_query
        return None, B_video_query

    # def encode_context(self, frame_video_feat, video_mask=None):
    #     # frame_video_feat = F.relu(self.vid_test_mapping(frame_video_feat))
    #     # # frame_video_feat = F.relu(self.test_mapping_linear1(frame_video_feat))
    #     # encoded_frame_feat_2 = self.vid_test_mapping_linear1(frame_video_feat)
    #
    #     encoded_frame_feat_2 = self.img_encoder(frame_video_feat)
    #     return None, encoded_frame_feat_2


    def encode_context(self, frame_video_feat, video_mask=None):
        B_encoded_frame_feat = self.encode_input(frame_video_feat, video_mask, self.B_frame_input_proj,
                                                 self.B_frame_encoder, self.B_frame_pos_embed)
        B_encoded_frame_feat = self.B_mapping_linear(B_encoded_frame_feat)
        if self.double_branch:
            A_encoded_frame_feat = self.encode_input(frame_video_feat, video_mask, self.A_frame_input_proj,
                                                     self.A_frame_encoder, self.A_frame_pos_embed)
            A_encoded_frame_feat = self.A_mapping_linear(A_encoded_frame_feat)
            return A_encoded_frame_feat, B_encoded_frame_feat
        return None, B_encoded_frame_feat

    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer):
        """
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            pos_embed_layer: positional embedding layer
        """
        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
        return encoder_layer(feat, mask)  # (N, L, D_hidden)

    def get_modularized_queries(self, encoded_query, query_mask,t=True):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        if t==True:
            modular_attention_scores = self.A_modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
        else:
            modular_attention_scores = self.B_modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)
        return modular_queries

    @staticmethod
    def get_clip_scale_scores(modularied_query, context_feat, mask=None):
        """ Calculate video2query scores for each pair of video and query inside the batch.
        Args:
            modularied_query: (N, D)
            context_feat: (N, L, D), output of the first transformer encoder layer
            context_mask: (N, L)
        Returns:
            context_query_scores: (N, N)  score of each query w.r.t. each video inside the batch,
                diagonal positions are positive. used to get negative samples.
        """
        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat = F.normalize(context_feat, dim=-1)
        if mask is None:
            clip_level_query_context_scores = torch.einsum("md,nld->mln", modularied_query, context_feat)  # (N, L, N)
        else:
            clip_level_query_context_scores = torch.einsum("md,nld->mln", modularied_query, context_feat)
            mask = mask.transpose(0, 1).unsqueeze(0)
            clip_level_query_context_scores = mask_logits(clip_level_query_context_scores, mask)

        query_context_scores, indices = torch.max(clip_level_query_context_scores,
                                                  dim=1)  # (N, N) diagonal positions are positive pairs
        return query_context_scores, clip_level_query_context_scores

    @staticmethod
    def get_unnormalized_clip_scale_scores(modularied_query, context_feat, mask=None):
        """ Calculate video2query scores for each pair of video and query inside the batch.
        Args:
            modularied_query: (N, D)
            context_feat: (N, L, D), output of the first transformer encoder layer
            context_mask: (N, L)
        Returns:
            context_query_scores: (N, N)  score of each query w.r.t. each video inside the batch,
                diagonal positions are positive. used to get negative samples.
        """

        if mask is None:
            query_context_scores = torch.einsum("md,nld->mln", modularied_query, context_feat)
        else:
            query_context_scores = torch.einsum("md,nld->mln", modularied_query, context_feat)
            mask = mask.transpose(0, 1).unsqueeze(0)
            query_context_scores = mask_logits(query_context_scores, mask)
        query_context_scores, _ = torch.max(query_context_scores, dim=1)
        return query_context_scores

    #原来的方法
    def get_pred_from_raw_query(self, video_query, video_query_B, video_feat_1=None, video_feat_2=None,
                                video_feat_mask=None,return_query_feats=False):
        if not return_query_feats:
            video_query, video_query_B = self.encode_query(video_query, video_query_B)
        if video_query is None:
            video_query_B = video_query_B.squeeze(1)
            max_predict_video_score, max_predict_video_score_ = self.get_clip_scale_scores(video_query_B,video_feat_2, video_feat_mask)
            if return_query_feats:
                # max_predict_video_score_ = self.get_unnormalized_clip_scale_scores(video_query_B, video_feat_2, video_feat_mask)
                return None, None, max_predict_video_score, max_predict_video_score_
            else:
                return max_predict_video_score, max_predict_video_score
        else:
            video_query = video_query.squeeze(1)
            video_query_B = video_query_B.squeeze(1)
            max_video_score, _ = self.get_clip_scale_scores(video_query, video_feat_1, video_feat_mask)
            max_predict_video_score, max_predict_video_score_ = self.get_clip_scale_scores(video_query_B,video_feat_2,video_feat_mask)
            if return_query_feats:
                max_video_score_ = self.get_unnormalized_clip_scale_scores(video_query, video_feat_1,video_feat_mask)
                # max_predict_video_score_ = self.get_unnormalized_clip_scale_scores(video_query_B, video_feat_2, video_feat_mask)
                return max_video_score, max_video_score_, max_predict_video_score, max_predict_video_score_
            else:
                return max_video_score, max_predict_video_score

    def get_clip_triplet_loss(self, query_context_scores, labels):
        v2t_scores = query_context_scores.t()
        t2v_scores = query_context_scores
        labels = np.array(labels)

        # cal_v2t_loss
        v2t_loss = 0
        for i in range(v2t_scores.shape[0]):
            pos_pair_scores = torch.mean(v2t_scores[i][np.where(labels == i)])
            neg_pair_scores, _ = torch.sort(v2t_scores[i][np.where(labels != i)[0]], descending=True)
            if self.config.use_hard_negative:
                sample_neg_pair_scores = neg_pair_scores[0]
            else:
                v2t_sample_max_idx = neg_pair_scores.shape[0]
                sample_neg_pair_scores = neg_pair_scores[
                    torch.randint(0, v2t_sample_max_idx, size=(1,)).to(v2t_scores.device)]
            v2t_loss += (self.config.margin + sample_neg_pair_scores - pos_pair_scores).clamp(min=0).sum()

        # cal_t2v_loss
        text_indices = torch.arange(t2v_scores.shape[0]).to(t2v_scores.device)
        t2v_pos_scores = t2v_scores[text_indices, labels]
        mask_score = copy.deepcopy(t2v_scores.data)
        mask_score[text_indices, labels] = 999
        _, sorted_scores_indices = torch.sort(mask_score, descending=True, dim=1)
        t2v_sample_max_idx = min(1 + self.config.hard_pool_size,
                                 t2v_scores.shape[1]) if self.config.use_hard_negative else t2v_scores.shape[1]
        sample_indices = sorted_scores_indices[
            text_indices, torch.randint(1, t2v_sample_max_idx, size=(t2v_scores.shape[0],)).to(t2v_scores.device)]
        # sample_indices = sorted_scores_indices[
        #     text_indices, torch.randint(1, 2, size=(t2v_scores.shape[0],)).to(t2v_scores.device)]
        t2v_neg_scores = t2v_scores[text_indices, sample_indices]

        t2v_loss = (self.config.margin + t2v_neg_scores - t2v_pos_scores).clamp(min=0)

        return t2v_loss.sum() / len(t2v_scores) + v2t_loss / len(v2t_scores)

    def get_frame_trip_loss(self, query_context_scores):
        """ ranking loss between (pos. query + pos. video) and (pos. query + neg. video) or (neg. query + pos. video)
        Args:
            query_context_scores: (N, N), cosine similarity [-1, 1],
                Each row contains the scores between the query to each of the videos inside the batch.
        """

        bsz = len(query_context_scores)

        diagonal_indices = torch.arange(bsz).to(query_context_scores.device)
        pos_scores = query_context_scores[diagonal_indices, diagonal_indices]  # (N, )
        query_context_scores_masked = copy.deepcopy(query_context_scores.data)
        # impossibly large for cosine similarity, the copy is created as modifying the original will cause error
        query_context_scores_masked[diagonal_indices, diagonal_indices] = 999
        pos_query_neg_context_scores = self.get_neg_scores(query_context_scores, query_context_scores_masked)
        neg_query_pos_context_scores = self.get_neg_scores(query_context_scores.transpose(0, 1),
                                                           query_context_scores_masked.transpose(0, 1))
        loss_neg_ctx = self.get_ranking_loss(pos_scores, pos_query_neg_context_scores)
        loss_neg_q = self.get_ranking_loss(pos_scores, neg_query_pos_context_scores)
        return loss_neg_ctx + loss_neg_q

    def get_neg_scores(self, scores, scores_masked):
        """
        scores: (N, N), cosine similarity [-1, 1],
            Each row are scores: query --> all videos. Transposed version: video --> all queries.
        scores_masked: (N, N) the same as scores, except that the diagonal (positive) positions
            are masked with a large value.
        """

        bsz = len(scores)
        batch_indices = torch.arange(bsz).to(scores.device)

        _, sorted_scores_indices = torch.sort(scores_masked, descending=True, dim=1)

        sample_min_idx = 1  # skip the masked positive

        sample_max_idx = min(sample_min_idx + self.config.hard_pool_size, bsz) if self.config.use_hard_negative else bsz

        # sample_max_idx = 2

        # (N, )
        sampled_neg_score_indices = sorted_scores_indices[batch_indices, torch.randint(sample_min_idx, sample_max_idx,
                                                                                       size=(bsz,)).to(scores.device)]

        sampled_neg_scores = scores[batch_indices, sampled_neg_score_indices]  # (N, )
        return sampled_neg_scores

    def get_ranking_loss(self, pos_score, neg_score):
        """ Note here we encourage positive scores to be larger than negative scores.
        Args:
            pos_score: (N, ), torch.float32
            neg_score: (N, ), torch.float32
        """
        return torch.clamp(self.config.margin + neg_score - pos_score, min=0).sum() / len(pos_score)

def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)
