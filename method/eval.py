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
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from method.model import DLDKD
from method.data_provider import Dataset4DLDKD,VisDataSet4DLDKD,\
    TxtDataSet4DLDKD,read_video_ids, collate_frame_val, collate_text_val
from utils.basic_utils import AverageMeter, BigFile, BigFile16, read_dict
from method.config import TestOptions
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# os.environ['CUDA_VISIBLE_DEVICES']='2'
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
    context_dataloader = DataLoader(eval_dataset, collate_fn=collate_frame_val, batch_size=opt.eval_context_bsz,
                                    num_workers=opt.num_workers, shuffle=False, pin_memory=opt.pin_memory)
    metas = []  # list(dicts)
    inher_frame_feat, explore_frame_feat, teacher_frame_feat, frame_mask = [],[],[],[]
    for idx, batch in tqdm(enumerate(context_dataloader), desc="Computing query2video scores",
                           total=len(context_dataloader)):
        metas.extend(batch[-1])
        frame_video_feat_ = batch[0].to(opt.device)
        frame_mask_ = batch[-3].to(opt.device)
        if len(batch) == 5:
            teacher_video_feat_ = batch[1].to(opt.device)
            _inher_frame_feat, _explore_frame_feat, _teacher_frame_feat = model.encode_context(frame_video_feat_, teacher_video_feat_, frame_mask_)
            teacher_frame_feat.append(_teacher_frame_feat)
        else:
            _inher_frame_feat, _explore_frame_feat  = model.encode_context(frame_video_feat_,frame_mask_)
        inher_frame_feat.append(_inher_frame_feat)
        explore_frame_feat.append(_explore_frame_feat)
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
    inher_frame_feat = cat_tensor(inher_frame_feat)
    frame_mask = cat_tensor(frame_mask)

    if model.double_branch:
        explore_frame_feat = cat_tensor(explore_frame_feat)
    else:
        explore_frame_feat = None

    if len(teacher_frame_feat) > 0:
        teacher_frame_feat = cat_tensor(teacher_frame_feat)
    else:
        teacher_frame_feat = None

    return dict(
        video_metas=metas,  # list(dict) (N_videos)
        inher_frame_feat=inher_frame_feat,
        explore_frame_feat=explore_frame_feat,
        teacher_frame_feat=teacher_frame_feat,
        video_mask=frame_mask
        )

def compute_query2ctx_info(model, eval_dataset, opt, ctx_info):
    model.eval()

    query_eval_loader = DataLoader(eval_dataset, collate_fn=collate_text_val, batch_size=opt.eval_query_bsz,
                                   num_workers=opt.num_workers, shuffle=False, pin_memory=opt.pin_memory)

    query_metas = []
    teacher_scores = []
    inher_scores = []
    explore_scores = []

    for idx, batch in tqdm(enumerate(query_eval_loader), desc="Computing q embedding", total=len(query_eval_loader)):

        query_metas.extend(batch[-1])
        query_feat = batch[0].to(opt.device)
        query_mask = batch[-3].to(opt.device)

        if ctx_info['teacher_frame_feat'] is not None:
            teacher_query = batch[1].to(opt.device)
            _teacher_scores, _ = \
                model.get_sim_scores(teacher_query,ctx_info['teacher_frame_feat'],ctx_info['video_mask'])
            teacher_scores.append(_teacher_scores)

        inheritance_query, exploration_query \
            = model.encode_query(query_feat, query_mask)
        _inher_scores, _ = \
            model.get_sim_scores(inheritance_query,ctx_info['inher_frame_feat'],ctx_info['video_mask'])

        inher_scores.append(_inher_scores)
        if model.double_branch:
            _explore_scores, _ = model.get_sim_scores(exploration_query, ctx_info['explore_frame_feat'], ctx_info['video_mask'])
            explore_scores.append(_explore_scores)

    inher_scores = torch.cat(inher_scores, dim=0).cpu().numpy().copy()
    if model.double_branch:
        explore_scores = torch.cat(explore_scores, dim=0).cpu().numpy().copy()
    else:
        explore_scores = None
    if len(teacher_scores) > 0:
        teacher_scores = torch.cat(teacher_scores, dim=0).cpu().numpy().copy()
        return inher_scores, explore_scores, teacher_scores, query_metas

    return inher_scores, explore_scores, None, query_metas



def cal_perf(t2v_all_errors, t2v_gt, test=False):

    # video retrieval
    (t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr) = eval_q2m(t2v_all_errors, t2v_gt)
    t2v_map_score = t2v_map(t2v_all_errors, t2v_gt)
    logging.info(" * Text to Video:")
    logging.info(" * r_1_5_10_100: {}".format([round(t2v_r1, 1), round(t2v_r5, 1), round(t2v_r10, 1), round(t2v_r100, 1)]))
    logging.info(" * recall sum: {}".format(round(t2v_r1+t2v_r5+t2v_r10+t2v_r100, 1)))
    logging.info(" * mAP: {}".format(round(t2v_map_score, 4)))
    logging.info(" * "+'-'*10)

    return (t2v_r1, t2v_r5, t2v_r10,t2v_r100, t2v_medr, t2v_meanr, t2v_map_score)


def eval_epoch(model, val_video_dataset, val_text_dataset, opt, test=False):
    """max_after_nms: always set to 100, since the eval script only evaluate top-100"""
    model.eval()
    logger.info("Computing scores")

    context_info = compute_context_info(model, val_video_dataset, opt)
    inher_scores, explore_scores, teacher_scores, query_metas = compute_query2ctx_info(model,val_text_dataset,opt,context_info)
    video_metas = context_info['video_metas']

    v2t_gt, t2v_gt = get_gt(video_metas, query_metas)
    
    if opt.double_branch:
        logging.info('inher_scores:')
        cal_perf(-1 * inher_scores, t2v_gt, test)
        logging.info('explore_scores:')
        cal_perf(-1 * explore_scores, t2v_gt,test)
        logging.info('score_sum:')
        score_sum = 0.7 * inher_scores + 0.3 * explore_scores
        t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr, t2v_map_score = cal_perf(-1 * score_sum, t2v_gt, test)
    else:
        t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr, t2v_map_score = cal_perf(-1 * inher_scores, t2v_gt,test)

    currscore = 0
    currscore += (t2v_r1 + t2v_r5 + t2v_r10 + t2v_r100)
    

    return currscore


def setup_model(opt):
    """Load model from checkpoint and move to specified device"""
    ckpt_filepath = os.path.join(opt.ckpt_filepath)
    checkpoint = torch.load(ckpt_filepath)
    loaded_model_cfg = checkpoint["model_cfg"]
    NAME_TO_MODELS = {'DLDKD':DLDKD}
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

def start_inference(opt=None):
    logger.info("Setup config, data and model...")
    if opt is None:
        opt = TestOptions().parse()
    cudnn.benchmark = False
    cudnn.deterministic = True

    rootpath = opt.root_path
    collection = opt.collection
    testCollection = '%stest' % collection
    trainCollection = '%strain' % collection
    cap_file = {'test': '%s.caption.txt' % testCollection,
                'train':'%s.caption.txt' % trainCollection}
    
    # caption
    caption_files = {x: os.path.join(rootpath, collection, 'TextData', cap_file[x])
                     for x in cap_file}

    text_feat_path = os.path.join(rootpath, collection, 'TextData', 'roberta_%s_query_feat.hdf5' % collection)
    # Load visual features
    visual_feat_path = os.path.join(rootpath, collection, 'FeatureData', opt.visual_feature)

    visual_feats = BigFile(visual_feat_path)
    video2frames =  read_dict(os.path.join(rootpath, collection, 'FeatureData', opt.visual_feature, 'video2frames.txt'))


    test_video_ids_list = read_video_ids(caption_files['test'])

    test_vid_dataset = VisDataSet4DLDKD(visual_feats, video2frames, opt,
                                               video_ids=test_video_ids_list)

    test_text_dataset = TxtDataSet4DLDKD(caption_files['test'], text_feat_path,opt)

    model = setup_model(opt)

    logger.info("Starting inference...")
    with torch.no_grad():
        score = eval_epoch(model, test_vid_dataset, test_text_dataset, opt, test=True)



if __name__ == '__main__':
    start_inference()