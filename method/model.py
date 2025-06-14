import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
from method.model_components import BertAttention, LinearLayer, TrainablePositionalEncoding
from method.model_components import clip_nce,clip_kl_only_pos, clip_nce_soft



    
class DLDKD(nn.Module):
    def __init__(self, config, opt):
        super(DLDKD, self).__init__()
        self.config = config
        self.double_branch = opt.double_branch

        # inheritance query encoder
        self.query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l,
                                                             hidden_size=config.inheritance_hidden,
                                                             dropout=config.input_drop)
        self.query_input_proj = LinearLayer(config.query_input_size, config.inheritance_hidden, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.query_encoder = BertAttention(edict(hidden_size=config.inheritance_hidden,
                                                intermediate_size=config.inheritance_hidden,
                                                hidden_dropout_prob=config.drop,
                                                num_attention_heads=config.n_heads,
                                                attention_probs_dropout_prob=config.drop))
        self.modular_vector_mapping = nn.Linear(config.inheritance_hidden, out_features=1, bias=False)

        # inheritance visual encoder
        self.visual_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                            hidden_size=config.inheritance_hidden,
                                                            dropout=config.input_drop)
        self.visual_input_proj = LinearLayer(config.visual_input_size, config.inheritance_hidden,
                                                   layer_norm=True,dropout=config.input_drop, relu=True)
        self.visual_encoder = copy.deepcopy(self.query_encoder)
        self.out_mapping_linear = nn.Linear(config.inheritance_hidden, config.inheritance_hidden)

        # exploration
        if self.double_branch:
            self.exp_query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l,
                                                               hidden_size=config.exploration_hidden,
                                                               dropout=config.input_drop)
            self.exp_query_input_proj = LinearLayer(config.query_input_size, config.exploration_hidden, layer_norm=True,
                                                dropout=config.input_drop, relu=True)
            self.exp_query_encoder = BertAttention(edict(hidden_size=config.exploration_hidden,
                                                     intermediate_size=config.exploration_hidden,
                                                     hidden_dropout_prob=config.drop,
                                                     num_attention_heads=config.n_heads,
                                                     attention_probs_dropout_prob=config.drop))
            self.exp_modular_vector_mapping = nn.Linear(config.exploration_hidden, out_features=1, bias=False)

            self.exp_visual_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                                hidden_size=config.exploration_hidden,
                                                                dropout=config.input_drop)
            self.exp_visual_input_proj = LinearLayer(config.visual_input_size, config.exploration_hidden,
                                                 layer_norm=True, dropout=config.input_drop, relu=True)
            self.exp_visual_encoder = copy.deepcopy(self.exp_query_encoder)
            self.exp_out_mapping_linear = nn.Linear(config.exploration_hidden, config.exploration_hidden)

        self.nce_criterion = clip_nce(reduction='mean')
        self.nce_criterion_soft = clip_nce_soft(reduction='mean')
        self.distill_loss = clip_kl_only_pos()

        self.weight = 1

        self.kl_intra_weight = opt.kl_intra_weight
        self.inher_nce_weight = opt.inher_nce_weight
        self.explore_nce_weight = opt.explore_nce_weight
        self.collection = opt.collection
        
        self.alpha = opt.alpha 
        self.belta = opt.belta

        self.reset_parameters()


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

    def forward(self, batch):
        label_dict = {}
        for index, label in enumerate(batch['text_labels']):
            if label in label_dict:
                label_dict[label].append(index)
            else:
                label_dict[label] = []
                label_dict[label].append(index)

        inheritance_encoded_feat, exploration_encoded_feat = self.encode_context(batch['student_videos'],batch['student_videos_mask'])
        inheritance_query, exploration_query = self.encode_query(batch['student_text'], batch['student_text_mask'])

        #teacher scores
        max_teacher_frame_scores, teacher_frame_scores \
            = self.get_sim_scores(batch['teacher_text'].squeeze(),batch['teacher_videos'],batch['student_videos_mask'])
        max_teacher_frame_scores_ \
            = self.get_unnormalized_sim_scores(batch['teacher_text'].squeeze(), batch['teacher_videos'],batch['student_videos_mask'])
        
        # inher scores
        max_inher_frame_scores, inher_frame_scores \
            = self.get_sim_scores(inheritance_query,inheritance_encoded_feat,batch['student_videos_mask'])
        max_inher_frame_scores_ \
            = self.get_unnormalized_sim_scores(inheritance_query, inheritance_encoded_feat,batch['student_videos_mask'])
        
        if self.double_branch:
            # explore scores
            max_explor_frame_scores, explor_frame_scores \
                = self.get_sim_scores(exploration_query,exploration_encoded_feat,batch['student_videos_mask'])
            max_explor_frame_scores_ \
                = self.get_unnormalized_sim_scores(exploration_query,exploration_encoded_feat,batch['student_videos_mask'])

        #loss
        inher_trip = 0
        inher_nce = 0
        kl = 0
        kl_intra = 0
        
        inher_trip = self.get_clip_triplet_loss(max_inher_frame_scores, batch['text_labels'])
        if self.config.label_style == 'soft':
            inher_nce = self.inher_nce_weight * self.nce_criterion_soft(batch['text_labels'],label_dict,max_inher_frame_scores_,
                                                                            max_teacher_frame_scores_,self.alpha,self.belta)
        else:
            inher_nce = self.inher_nce_weight * self.nce_criterion(batch['text_labels'],label_dict,max_inher_frame_scores_)
     
        explore_trip = 0
        explore_nce = 0
        if self.double_branch:
            explore_trip = self.get_clip_triplet_loss(max_explor_frame_scores, batch['text_labels'])
            if self.config.label_style == 'soft':
                explore_nce = self.explore_nce_weight * self.nce_criterion_soft(batch['text_labels'],label_dict,max_explor_frame_scores_,
                                                                            max_explor_frame_scores_,self.alpha,self.belta)
            else:
                explore_nce = self.explore_nce_weight * self.nce_criterion(batch['text_labels'],label_dict,max_explor_frame_scores_)

        kl_intra = self.kl_intra_weight * self.weight * self.compute_kl_loss(inher_frame_scores, teacher_frame_scores, batch['student_videos_mask'],
                                       0.2,mode='frame_score',query_labels=batch['text_labels'])
        kl =  kl_intra 

        loss = inher_trip + inher_nce + kl + explore_trip + explore_nce

        return loss, {"loss_overall": float(loss), 'inher_trip': inher_trip,
                      'inher_nce': inher_nce, 'explore_trip': explore_trip,
                      'explore_nce':explore_nce ,'kl':kl,  'kl_intra':kl_intra
                      }
       

    def compute_kl_loss(self, predict, target, cnn_mask, temp, mode='batch_score', query_labels=None):
        if mode == 'batch_score':
            # t2v
            t2v_kl_loss = 0
            t2v_predict = F.log_softmax(predict / temp, dim=-1)
            t2v_target = F.softmax(target / temp, dim=-1)
            t2v_kl_loss = F.kl_div(t2v_predict, t2v_target, reduction='batchmean')

            # v2t
            v2t_kl_loss = 0
            v2t_predict = F.log_softmax(predict.t() / temp, dim=-1)
            v2t_target = F.softmax(target.t() / temp, dim=-1)
            v2t_kl_loss = F.kl_div(v2t_predict, v2t_target, reduction='batchmean')

            kl_loss = t2v_kl_loss + v2t_kl_loss

            return kl_loss
        else:
            predict = [predict[i, :, x] for i, x in enumerate(query_labels)]
            predict = torch.stack(predict)
            target = [target[i, :, x] for i, x in enumerate(query_labels)]
            target = torch.stack(target)

            cnn_mask = cnn_mask[query_labels]
            kl_loss = 0
            for i in range(predict.shape[0]):
                feat_len = torch.nonzero(cnn_mask[i] > 0).shape[0]
                predict_ = F.log_softmax(predict[i, :feat_len] / temp, dim=-1)
                target_ = F.softmax(target[i, :feat_len] / temp, dim=-1)
                kl_loss += F.kl_div(predict_, target_, reduction='sum')

            return kl_loss

    def encode_query(self, query_feat, query_mask):
        inheritance_query = self.encode_input(query_feat, query_mask, self.query_input_proj,
                                              self.query_encoder, self.query_pos_embed)  # (N, Lq, D)

        inheritance_query = self.get_modularized_queries(inheritance_query, query_mask, True)  # (N, D) * 1

        if self.double_branch:
            exploration_query = self.encode_input(query_feat, query_mask, self.exp_query_input_proj,
                                                self.exp_query_encoder,self.exp_query_pos_embed)
            exploration_query = self.get_modularized_queries(exploration_query, query_mask)  # (N, D) * 1
            return inheritance_query, exploration_query

        return inheritance_query, None

    

    def encode_context(self, frame_video_feat, video_mask=None):

        inheritance_encoded_feat = self.encode_input(frame_video_feat, video_mask, self.visual_input_proj,
                                                 self.visual_encoder, self.visual_pos_embed)
        inheritance_encoded_feat = self.out_mapping_linear(inheritance_encoded_feat)

        if self.double_branch:
            exploration_encoded_feat = self.encode_input(frame_video_feat, video_mask, self.exp_visual_input_proj,
                                                     self.exp_visual_encoder, self.exp_visual_pos_embed)
            exploration_encoded_feat = self.exp_out_mapping_linear(exploration_encoded_feat)
            return inheritance_encoded_feat, exploration_encoded_feat

        return inheritance_encoded_feat, None

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

    def get_modularized_queries(self, encoded_query, query_mask, inheritance=False):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        if inheritance:
            modular_attention_scores = self.modular_vector_mapping(encoded_query)
        else:
            modular_attention_scores = self.exp_modular_vector_mapping(encoded_query)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)
        return modular_queries.squeeze()
    @staticmethod
    def get_query_sim_scores(modularied_query):
        """
        Calculate query2query scores for each pair of queries inside the batch.
        Args:
            modularied_query: (N, D)
        Returns:
            query_query_scores: (N, N) score of each query w.r.t. each other query inside the batch,
                                diagonal positions are self-similarities (should be 1).
        """

        modularied_query = F.normalize(modularied_query, dim=-1)
        
 
        query_query_scores = torch.einsum("nd,md->nm", modularied_query, modularied_query)
        
        return query_query_scores
    @staticmethod
    def get_video_sim_scores(video_tensor, mode="max"):
        """
        Calculate video-to-video similarity scores based on the maximum frame similarity.
        
        Args:
            video_tensor: (batch, frames, dim) tensor, where batch is the number of videos,
                        frames is the number of frames in each video, and dim is the feature dimension.
            mode: max== Take the maximum frame similarity between videos as the video similarity
                  mean== First, take the average of the video frames to get the video representation, and then calculate the similarity between videos
        Returns:
            video_video_scores: (batch, batch) similarity score between each pair of videos,
                                where the score is the maximum frame similarity.
        """
       
        video_tensor = F.normalize(video_tensor, dim=-1)
        if mode == "max":
           
            sim_matrix = torch.einsum('bfd,kfd->bkf', video_tensor, video_tensor)  # (batch, batch, frames)
            
           
            video_video_scores = sim_matrix.max(dim=-1).values
        elif mode == "mean":
            video_tensor = torch.mean(video_tensor, dim=1)
           
            video_video_scores = torch.einsum('bd,kd->bk', video_tensor, video_tensor)  # (batch, batch, frames)
            
            
        return video_video_scores


    @staticmethod
    def get_sim_scores(modularied_query, context_feat, mask=None):
        """ Calculate video2query scores for each pair of video and query inside the batch. cosine sim
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
    def get_unnormalized_sim_scores(modularied_query, context_feat, mask=None):
        """ Calculate video2query scores for each pair of video and query inside the batch. 向量点积值
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
