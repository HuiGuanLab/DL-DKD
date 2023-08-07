import numpy as np
import logging
import torch.backends.cudnn as cudnn
import torch
import h5py
import os
from tqdm import tqdm
import pickle
from collections import defaultdict
from torch.utils.data import DataLoader


from method.model import MS_SL_Net
from method.data_provider import Dataset4MS_SL,VisDataSet4MS_SL,\
    TxtDataSet4MS_SL,read_video_ids, collate_frame_val, collate_text_val
from utils.basic_utils import AverageMeter, BigFile, read_dict
from method.config import TestOptions
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def ap_score(sorted_labels):
    nr_relevant = len([x for x in sorted_labels if x > 0])
    if nr_relevant == 0:
        return 0.0

    length = len(sorted_labels)
    ap = 0.0
    rel = 0

    for i in range(length):
        lab = sorted_labels[i]
        if lab >= 1:
            rel += 1
            ap += float(rel) / (i + 1.0)
    ap /= nr_relevant
    return ap

def get_gt(video_metas, query_metas):
    v2t_gt = []
    for vid_id in video_metas:
        v2t_gt.append([])
        for i, query_id in enumerate(query_metas):
            if query_id.split('#', 1)[0] == vid_id:
                v2t_gt[-1].append(i)

    t2v_gt = {}
    for i, t_gts in enumerate(v2t_gt):
        for t_gt in t_gts:
            t2v_gt.setdefault(t_gt, [])
            t2v_gt[t_gt].append(i)

    return v2t_gt, t2v_gt

def eval_q2m(scores, q2m_gts):
    '''
    Image -> Text / Text -> Image
    Args:
      scores: (n_query, n_memory) matrix of similarity scores
      q2m_gts: list, each item is the positive memory ids of the query id
    Returns:
      scores: (recall@1, 5, 10, median rank, mean rank)
      gt_ranks: the best ranking of ground-truth memories
    '''
    n_q, n_m = scores.shape

    gt_ranks = np.zeros((n_q,), np.int32)
    aps = np.zeros(n_q)
    for i in range(n_q):
        s = scores[i]
        sorted_idxs = np.argsort(s)
        rank = n_m + 1
        tmp_set = []
        for k in q2m_gts[i]:
            tmp = np.where(sorted_idxs == k)[0][0] + 1
            if tmp < rank:
                rank = tmp

        gt_ranks[i] = rank

    # compute metrics
    r1 = 100.0 * len(np.where(gt_ranks <= 1)[0]) / n_q
    r5 = 100.0 * len(np.where(gt_ranks <= 5)[0]) / n_q
    r10 = 100.0 * len(np.where(gt_ranks <= 10)[0]) / n_q
    r100 = 100.0 * len(np.where(gt_ranks <= 100)[0]) / n_q
    medr = np.median(gt_ranks)
    meanr = gt_ranks.mean()
    # mAP = aps.mean()

    return (r1, r5, r10, r100, medr, meanr)

# mAP for Text-to-Video Retrieval
def t2v_map(c2i, t2v_gts):
    """
    Text->Videos (Text-to-Video Retrieval)
    c2i: (5N, N) matrix of caption to video errors
    """
    perf_list = []
    for i in range(c2i.shape[0]):
        d_i = c2i[i, :]
        labels = [0]*len(d_i)

        x = t2v_gts[i][0]
        labels[x] = 1

        sorted_labels = [labels[x] for x in np.argsort(d_i)]

        current_score = ap_score(sorted_labels)
        perf_list.append(current_score)
    return np.mean(perf_list)


def compute_context_info(model, eval_dataset, opt):
    """Use val set to do evaluation, remember to run with torch.no_grad().
    estimated 2200 (videos) * 100 (frm) * 500 (hsz) * 4 (B) * 2 (video/sub) * 2 (layers) / (1024 ** 2) = 1.76 GB
    max_n_videos: only consider max_n_videos videos for each query to return st_ed scores.
    """
    model.eval()
    n_total_vid = len(eval_dataset)
    context_dataloader = DataLoader(eval_dataset, collate_fn=collate_frame_val, batch_size=opt.eval_context_bsz,
                                    num_workers=opt.num_workers, shuffle=False, pin_memory=opt.pin_memory)
    bsz = opt.eval_context_bsz
    metas = []  # list(dicts)
    frame_feat, clip_feat, frame_mask = [], [], []
    for idx, batch in tqdm(enumerate(context_dataloader), desc="Computing query2video scores",
                           total=len(context_dataloader)):
        metas.extend(batch[-1])
        frame_video_feat_ = batch[0].to(opt.device)
        frame_mask_ = batch[1].to(opt.device)
        _frame_feat, _clip_feat = model.encode_context(frame_video_feat_, frame_mask_)
        if _frame_feat == None:
            _frame_feat = _clip_feat
        frame_feat.append(_frame_feat)
        clip_feat.append(_clip_feat)
        frame_mask.append(frame_mask_)

    def cat_tensor(tensor_list):
        if len(tensor_list) == 0:
            return None
        else:
            seq_l = [e.shape[1] for e in tensor_list]
            b_sizes = [e.shape[0] for e in tensor_list]
            b_sizes_cumsum = np.cumsum([0] + b_sizes)
            if len(tensor_list[0].shape) == 3:
                hsz = tensor_list[0].shape[2]
                res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l), hsz)
            elif len(tensor_list[0].shape) == 2:
                res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l))
            else:
                raise ValueError("Only support 2/3 dimensional tensors")
            for i, e in enumerate(tensor_list):
                res_tensor[b_sizes_cumsum[i]:b_sizes_cumsum[i+1], :seq_l[i]] = e
            return res_tensor

    return dict(
        video_metas=metas,  # list(dict) (N_videos)
        video_feat=cat_tensor(frame_feat),
        clip_feat=cat_tensor(clip_feat),
        video_mask=cat_tensor(frame_mask)
        )

def compute_query2ctx_info(model, eval_dataset, opt, ctx_info):
    model.eval()

    query_eval_loader = DataLoader(eval_dataset, collate_fn=collate_text_val, batch_size=opt.eval_query_bsz,
                                   num_workers=opt.num_workers, shuffle=False, pin_memory=opt.pin_memory)

    # n_total_query = len(eval_dataset)
    # bsz = opt.eval_query_bsz
    query_metas = []
    # query_context_scores = None
    scores = []
    clip_scores = []
    score_sum = []
    for idx, batch in tqdm(enumerate(query_eval_loader), desc="Computing q embedding", total=len(query_eval_loader)):

        _query_metas = batch[-1]
        query_metas.extend(batch[-1])
        query_feat = batch[0].to(opt.device)
        query_mask = batch[1].to(opt.device)
        _scores, _clip_scores = model.get_pred_from_raw_query(
            query_feat, query_mask, ctx_info["video_feat"], ctx_info["clip_feat"], ctx_info['video_mask'])
        # _score_sum = _scores + _clip_scores

        scores.append(_scores)
        clip_scores.append(_clip_scores)
        # score_sum.append(_score_sum)

    scores = torch.cat(scores, dim=0).cpu().numpy().copy()
    clip_scores = torch.cat(clip_scores, dim=0).cpu().numpy().copy()
    # score_sum = torch.cat(score_sum, dim=0).cpu().numpy().copy()
    return scores, clip_scores, score_sum, query_metas



def cal_perf(t2v_all_errors, t2v_gt, test=False):

    # video retrieval
    (t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr) = eval_q2m(t2v_all_errors, t2v_gt)
    t2v_map_score = t2v_map(t2v_all_errors, t2v_gt)
    logging.info(" * Text to Video:")
    logging.info(" * r_1_5_10_100, medr, meanr: {}".format([round(t2v_r1, 1), round(t2v_r5, 1), round(t2v_r10, 1), round(t2v_r100, 1)]))
    logging.info(" * recall sum: {}".format(round(t2v_r1+t2v_r5+t2v_r10+t2v_r100, 1)))
    logging.info(" * mAP: {}".format(round(t2v_map_score, 4)))
    if test:
        logging.info(" * r_1_5_10_100, mAP, sum: {}".format([round(t2v_r1, 1), round(t2v_r5, 1), round(t2v_r10, 1),
                                                            round(t2v_r100, 1), round(t2v_map_score, 4),
                                                            round(t2v_r1 + t2v_r5 + t2v_r10 + t2v_r100, 1)]))
    logging.info(" * "+'-'*10)

    return (t2v_r1, t2v_r5, t2v_r10,t2v_r100, t2v_medr, t2v_meanr, t2v_map_score)


def eval_epoch(model, val_video_dataset, val_text_dataset, opt, test=False):
    """max_after_nms: always set to 100, since the eval script only evaluate top-100"""
    model.eval()
    logger.info("Computing scores")

    context_info = compute_context_info(model, val_video_dataset, opt)
    scores, clip_scores, score_sum, query_metas = compute_query2ctx_info(model,val_text_dataset,opt,context_info)
    video_metas = context_info['video_metas']

    v2t_gt, t2v_gt = get_gt(video_metas, query_metas)

    logging.info('scores:')
    cal_perf(-1 * scores, t2v_gt,test)
    logging.info('clip_scores:')
    if opt.double_branch:
        cal_perf(-1 * clip_scores, t2v_gt,test)
    else:
        t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr, t2v_map_score = cal_perf(-1 * clip_scores, t2v_gt,test)
    logging.info('score_sum:')
    score_sum = opt.frame_weight * scores + opt.clip_weight * clip_scores
    if not opt.double_branch:
        cal_perf(-1 * score_sum, t2v_gt, test)
    if opt.double_branch:
        if test:
            for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                logging.info(f'{i}:{1-i}')
                if i==0.5:
                    t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr, t2v_map_score = cal_perf(-1 * (i * scores + (1 - i) * clip_scores), t2v_gt,test)
                else:
                    cal_perf(-1 * (i*scores+(1-i)*clip_scores), t2v_gt, test)
        else:
            t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr, t2v_map_score = cal_perf(-1 * score_sum, t2v_gt, test)

    currscore = 0
    currscore += (t2v_r1 + t2v_r5 + t2v_r10 + t2v_r100)
    if not test:
        score_sum = 0.3 * scores + 0.7 * clip_scores
        t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr, t2v_map_score = cal_perf(-1 * score_sum, t2v_gt, test)
        a = 0
        a += (t2v_r1 + t2v_r5 + t2v_r10 + t2v_r100)
        opt.writer.add_scalar("Eval/t2v_r1", t2v_r1, model.epoch)
        opt.writer.add_scalar("Eval/t2v_r5", t2v_r5, model.epoch)
        opt.writer.add_scalar("Eval/t2v_r10", t2v_r10, model.epoch)
        opt.writer.add_scalar("Eval/t2v_r100", t2v_r100, model.epoch)
        opt.writer.add_scalar("Eval/t2v_all", a, model.epoch)

    return currscore
def l2norm(X):
    """L2-normalize columns of X
    """
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return 1.0 * X / norm

import matplotlib.pyplot as plt
def uniform_feature_sampling(features, max_len):
    num_clips = features.shape[0]

    if max_len is None or num_clips <= max_len:
        return features
    idxs = np.arange(0, max_len + 1, 1.0) / max_len * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1
    new_features = []
    for i in range(max_len):
        s_idx, e_idx = idxs[i], idxs[i + 1]
        if s_idx < e_idx:
            new_features.append(np.mean(features[s_idx:e_idx], axis=0))
        else:
            new_features.append(features[s_idx])
    new_features = np.asarray(new_features)
    return new_features
def eval_dataset_clip(dataset):
    clip_vid_feat = h5py.File(os.path.join(f'/home/zms/VisualSearch/{dataset}/'
                                           f'FeatureData/new_clip_vit_32_{dataset}_vid_features.hdf5'), 'r')
    clip_txt_feat = h5py.File(
        os.path.join(f'/home/zms/VisualSearch/{dataset}/TextData/clip_ViT_B_32_{dataset}_query_feat.hdf5'), 'r')
    # clip_vid_feat = h5py.File(os.path.join(f'/home/zms/cxk/TCL/ac_tcl_vid_feat.hdf5'), 'r')
    # clip_txt_feat = h5py.File(
    #     os.path.join(f'/home/zms/cxk/TCL/ac_tcl_text_feat.hdf5'), 'r')
    video_id=[]
    video_feat=[]
    txt_feat = []
    txt_id = []
    with open(f'/home/zms/VisualSearch/{dataset}/TextData/{dataset}test.caption.txt','r') as reader:
        data=reader.readlines()
        for i in data:
            t_id,cap=i.split(' ',1)
            txt_id.append(t_id)
            txt_feat.append(np.array(clip_txt_feat[t_id]))
            v_id = t_id.split('#',1)[0]
            if v_id not in video_id:
                video_id.append(v_id)
                video_feat.append(np.array(clip_vid_feat[v_id]))

    txt_feat = np.stack(txt_feat)
    scores = np.zeros(shape=(len(txt_feat),len(video_feat)))
    txt_feat = l2norm(txt_feat)
    for i,feat in tqdm(enumerate(video_feat),total=len(video_feat)):
        # feat = np.mean(feat,axis=0,keepdims=True)
        feat = l2norm(feat)
        v_score = np.dot(txt_feat, feat.T) #150000,L
        v_score = np.max(v_score, axis=-1)
        # v_score = v_score.squeeze()
        scores[:,i] = v_score
    _, t = get_gt(video_id, txt_id)
    cal_perf(-1 * scores, t)

def eval_clip():
    clip_vid_feat = h5py.File(os.path.join('/home/zms/VisualSearch/tvr/'
                                           'FeatureData/new_clip_vit_32_tvr_test_features.hdf5'), 'r')
    clip_txt_feat = h5py.File(
        os.path.join('/home/zms/VisualSearch/tvr/TextData/clip_ViT_B_32_tvr_query_feat.hdf5'), 'r')

    video_id=[]
    video_feat=[]
    txt_feat = []
    txt_id = []
    score_min=1000
    score_max=-10000

    # video_feat = np.stack(video_feat)
    # print(video_feat.shape)
    #####只提取正样本之间的分数
    # total_score={}
    # for asd in [8,10,12,14,16,18,20]:
    #     with open('/home/zms/VisualSearch/activitynet/TextData/activitynettrain.caption.txt','r') as reader:
    #         data=reader.readlines()
    #         for i in tqdm(data):
    #             t_id,cap=i.split(' ',1)
    #             txt_id.append(t_id)
    #             text_feat = l2norm(np.array(clip_txt_feat[t_id]))
    #             v_id = t_id.split('#',1)[0]
    #             vid_feat = np.array(clip_vid_feat[v_id])
    #             vid_feat = l2norm(uniform_feature_sampling(vid_feat,128))
    #             v_score = np.dot(vid_feat, text_feat.T)
    #             for j in v_score:
    #                 j=j[0]
    #                 j=round(j,asd)
    #                 if j not in total_score.keys():
    #                     total_score[j] = 0
    #                 total_score[j] += 1
    #     x = []
    #     y = []
    #     total_score_key = sorted(total_score.keys(), reverse=False)
    #     for key in total_score_key:
    #         x.append(key)
    #         y.append(total_score[key])
    #     fig, ax = plt.subplots()
    #     ax.plot(x, y)
    #     plt.draw()
    #     plt.savefig(f"./only_Pos_{asd}points_Accuracy_rate.jpg")
    #
    # exit()
    # ######
    with open('/home/zms/VisualSearch/tvr/TextData/tvrtest.caption.txt','r') as reader:
        data=reader.readlines()
        for i in data:
            t_id,cap=i.split(' ',1)
            txt_id.append(t_id)
            txt_feat.append(np.array(clip_txt_feat[t_id]))
            v_id = t_id.split('#',1)[0]
            if v_id not in video_id:
                video_id.append(v_id)
                video_feat.append(np.array(clip_vid_feat[v_id]))

    txt_feat = np.stack(txt_feat).squeeze(1)
    scores = np.zeros(shape=(len(txt_feat),len(video_feat)))
    txt_feat = l2norm(txt_feat)
    # total_score={}
    for i,feat in tqdm(enumerate(video_feat),total=len(video_feat)):
        feat = l2norm(feat)
        v_score = np.dot(txt_feat, feat.T) #150000,L
        v_score = np.max(v_score,axis=-1)
        scores[:,i] = v_score
        # for i in v_score:
        #     for j in i:
        #         if j not in total_score.keys():
        #             total_score[j]=0
        #         total_score[j] += 1
        # fi_score = np.zeros(shape=(v_score.shape[0],))
        # index = np.argsort(v_score, axis=1)[:,-10:]
        # for idx, ii in enumerate(index):
        #     fi_score[idx] = np.sum(v_score[idx,ii],keepdims=True)
        # scores[:,i] = fi_score

    # x=[]
    # y=[]
    # total_score_key = sorted(total_score.keys(), reverse=False)
    # for key in total_score_key:
    #     x.append(key)
    #     y.append(total_score[key])
    #     if key > score_max:
    #         score_max = key
    #     if key < score_min:
    #         score_min = key

    # print(score_min)
    # print(score_max)
    # fig, ax = plt.subplots()
    # ax.plot(x, y)
    # plt.draw()
    # plt.savefig(f"./Accuracy rate.jpg")



def setup_model(opt):
    """Load model from checkpoint and move to specified device"""
    ckpt_filepath = os.path.join(opt.ckpt_filepath)
    checkpoint = torch.load(ckpt_filepath)
    loaded_model_cfg = checkpoint["model_cfg"]
    NAME_TO_MODELS = {'MS_SL_Net':MS_SL_Net}
    model = NAME_TO_MODELS[opt.model_name](loaded_model_cfg,opt)
    
    model.load_state_dict(checkpoint["model"])
    logger.info("Loaded model saved at epoch {} from checkpoint: {}".format(checkpoint["epoch"], opt.ckpt_filepath))

    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        if len(opt.device_ids) > 1:
            logger.info("Use multi GPU", opt.device_ids)
            model = torch.nn.DataParallel(model, device_ids=opt.device_ids)  # use multi GPU
    return model

def start_inference():
    logger.info("Setup config, data and model...")
    opt = TestOptions().parse()
    cudnn.benchmark = False
    cudnn.deterministic = True

    rootpath = opt.root_path
    collection = opt.collection
    testCollection = '%stest' % collection

    cap_file = {'test': '%s.caption.txt' % testCollection}

    # caption
    caption_files = {x: os.path.join(rootpath, collection, 'TextData', cap_file[x])
                     for x in cap_file}

    text_feat_path = os.path.join(rootpath, collection, 'TextData', 'roberta_%s_query_feat.hdf5' % collection)
    # Load visual features
    visual_feat_path = os.path.join(rootpath, collection, 'FeatureData', opt.visual_feature)

    visual_feats = BigFile(visual_feat_path)
    video2frames =  read_dict(os.path.join(rootpath, collection, 'FeatureData', opt.visual_feature, 'video2frames.txt'))

    test_video_ids_list = read_video_ids(caption_files['test'])
    test_vid_dataset = VisDataSet4MS_SL(visual_feats, video2frames, opt,
                                               video_ids=test_video_ids_list)

    test_text_dataset = TxtDataSet4MS_SL(caption_files['test'], text_feat_path,opt)

    model = setup_model(opt)

    logger.info("Starting inference...")
    with torch.no_grad():
        score = eval_epoch(model, test_vid_dataset, test_text_dataset, opt, test=True)



if __name__ == '__main__':
    eval_dataset_clip("tvr")
    exit()
    start_inference()