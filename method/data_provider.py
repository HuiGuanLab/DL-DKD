import json
import random

import torch
import torch.utils.data as data
import numpy as np
import re
import h5py
import os
from sklearn.cluster import DBSCAN

def getVideoId(cap_id):
    vid_id = cap_id.split('#')[0]
    return vid_id

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip().lower().split()

def read_video_ids(cap_file):
    video_ids_list = []
    with open(cap_file, 'r') as cap_reader:
        for line in cap_reader.readlines():
            cap_id, caption = line.strip().split(' ', 1)
            video_id = getVideoId(cap_id)
            if video_id not in video_ids_list:
                video_ids_list.append(video_id)
    return video_ids_list

def average_to_fixed_length(visual_input, map_size):
    visual_input = torch.from_numpy(visual_input)
    num_sample_clips = map_size
    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips + 1, 1.0) / num_sample_clips * num_clips

    idxs = torch.min(torch.round(idxs).long(), torch.tensor(num_clips - 1))

    new_visual_input = []

    for i in range(num_sample_clips):

        s_idx, e_idx = idxs[i].item(), idxs[i + 1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx], dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0).numpy()


    return new_visual_input

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


def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)

def cat_videos(video_features):
    bsz = len(video_features)
    hidden = video_features[0].shape[-1]
    video_lengths = [len(frame) for frame in video_features]
    frame_videos = torch.zeros(bsz, max(video_lengths), hidden)
    videos_mask = torch.zeros(bsz, max(video_lengths))
    for i, frames in enumerate(video_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0

    return frame_videos, videos_mask

def cat_captions(captions):
    feat_dim = captions[0][0].shape[-1]

    merge_captions = []
    all_lengths = []
    labels = []

    for index, caps in enumerate(captions):
        labels.extend(index for i in range(len(caps)))
        all_lengths.extend(len(cap) for cap in caps)
        merge_captions.extend(cap for cap in caps)

    target = torch.zeros(len(all_lengths), max(all_lengths), feat_dim)
    words_mask = torch.zeros(len(all_lengths), max(all_lengths))

    for index, cap in enumerate(merge_captions):
        end = all_lengths[index]
        target[index, :end, :] = cap[:end, :]
        words_mask[index, :end] = 1.0

    return target, words_mask, labels


def collate_train(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)
    student_video_features, captions, teacher_video_features, clip_captions, idxs, cap_ids, video_ids = zip(*data)

    # videos
    student_videos, student_videos_mask = cat_videos(student_video_features)
    teacher_videos, teacher_videos_mask = cat_videos(teacher_video_features)

    # captions
    student_target, student_text_mask, text_labels = cat_captions(captions)
    teacher_target, _, __ = cat_captions(clip_captions)


    return dict(student_videos=student_videos,
                teacher_videos=teacher_videos,
                student_videos_mask=student_videos_mask,
                student_text=student_target.squeeze(),
                student_text_mask=student_text_mask,
                teacher_text=teacher_target.squeeze(),
                text_labels=text_labels,
                )


def collate_frame_val(data):
    if len(data[0]) == 3:
        student_video_features, idxs, video_ids = zip(*data)
        student_videos, student_videos_mask = cat_videos(student_video_features)
        return student_videos, student_videos_mask, idxs, video_ids
    else:
        student_video_features, teacher_video_features, idxs, video_ids = zip(*data)
        student_videos, student_videos_mask = cat_videos(student_video_features)
        teacher_videos, _ = cat_videos(teacher_video_features)
        return student_videos, teacher_videos, student_videos_mask, idxs, video_ids



def collate_text_val(data):
    if data[0][0] is not None:
        data.sort(key=lambda x: len(x[0]), reverse=True)
    if len(data[0]) == 3:
        captions, idxs, cap_ids = zip(*data)
        teacher_target = None
    else:
        captions, teacher_captions, idxs, cap_ids = zip(*data)
        teacher_target = torch.cat(teacher_captions,dim=0)
    lengths = [len(cap) for cap in captions]
    target = torch.zeros(len(captions), max(lengths), captions[0].shape[-1])
    words_mask = torch.zeros(len(captions), max(lengths))
    for i, cap in enumerate(captions):
        end = lengths[i]
        target[i, :end] = cap[:end]
        words_mask[i, :end] = 1.0
    if teacher_target is not None:
        return target, teacher_target, words_mask, idxs, cap_ids
    return target, words_mask, idxs, cap_ids

class Dataset4DLDKD(data.Dataset):
    """
    Load captions and video frame features by pre-trained CNN model.
    """

    def __init__(self, cap_file, visual_feat, text_feat_path, clip_vid_feat_path,clip_text_feat_path,opt, video2frames=None):
        # Captions
        self.captions = {}
        self.cap_ids = []
        self.video_ids = []
        self.vid_caps = {}
        self.video2frames = video2frames
 
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                video_id = getVideoId(cap_id)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
                if video_id not in self.video_ids:
                    self.video_ids.append(video_id)
                if video_id in self.vid_caps:
                    self.vid_caps[video_id].append(cap_id)
                else:
                    self.vid_caps[video_id] = []
                    self.vid_caps[video_id].append(cap_id)

        self.max_ctx_len = opt.max_ctx_l
        self.max_desc_len = opt.max_desc_l
        self.length = len(self.vid_caps)

        self.teacher = opt.teacher
        self.student = opt.student
        self.visual_feat = visual_feat
        self.student_text_feat = h5py.File(text_feat_path, 'r')
        self.clip_text_feat = h5py.File(clip_text_feat_path, 'r')
        self.clip_vid_feat = h5py.File(clip_vid_feat_path, 'r')
       
        

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        cap_ids = self.vid_caps[video_id]

        # video
        if self.student == 'i3d':
            frame_list = self.video2frames[video_id]
            student_vecs = []
            for frame_id in frame_list:
                student_vecs.append(self.visual_feat.read_one(frame_id))
        else:
            student_vecs = self.student_vid_feat[video_id][:]

        


        teacher_vecs = self.clip_vid_feat[video_id][:]

        student_vecs = np.array(student_vecs)
        student_vecs = uniform_feature_sampling(student_vecs, teacher_vecs.shape[0])

        student_video_feature = uniform_feature_sampling(student_vecs, self.max_ctx_len)
        student_video_feature = torch.from_numpy(l2_normalize_np_array(student_video_feature))

        teacher_video_feature = uniform_feature_sampling(teacher_vecs, self.max_ctx_len)
        teacher_video_feature = torch.from_numpy(teacher_video_feature)
        # teacher_video_feature = torch.from_numpy(l2_normalize_np_array(teacher_video_feature))

        # text
        cap_tensors = []
        clip_cap_tensors = []
        for cap_id in cap_ids:
            
            cap_feat = self.student_text_feat[cap_id][...]
            cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat)).squeeze()[:self.max_desc_len]
            cap_tensors.append(cap_tensor)
            
            
            try:
                clip_cap_feat = self.clip_text_feat[cap_id][...]
            except KeyError as e:
                if "Unable to synchronously open object" in str(e):
                    clip_cap_feat = self.clip_text_feat['#'.join(cap_id.split('#enc#'))][...]
                else:
                   
                    raise e
            clip_cap_feat = torch.from_numpy(clip_cap_feat)
            # clip_cap_feat = torch.from_numpy(l2_normalize_np_array(clip_cap_feat))
            clip_cap_tensors.append(clip_cap_feat)


        return student_video_feature, cap_tensors, teacher_video_feature,clip_cap_tensors, index, cap_ids, video_id

    def __len__(self):
        return self.length

class VisDataSet4DLDKD(data.Dataset):

    def __init__(self, visual_feat, video2frames=None, opt=None, video_ids=None):
        self.visual_feat = visual_feat
        self.video2frames = video2frames
        if video_ids is not None:
            self.video_ids = video_ids
        else:
            self.video_ids = video2frames.keys()
        self.teacher_feat = None
        
        self.length = len(self.video_ids)
        self.max_ctx_len = opt.max_ctx_l
        self.student = opt.student
        
    def __getitem__(self, index):
        video_id = self.video_ids[index]
        if self.student == 'i3d':
            frame_list = self.video2frames[video_id]
            student_vecs = []
            for frame_id in frame_list:
                student_vecs.append(self.visual_feat.read_one(frame_id))
            student_vecs = np.array(student_vecs)
        else:
            student_vecs = self.visual_feat[video_id][:]

        if self.teacher_feat is not None:
            teacher_vecs = self.teacher_feat[video_id][:]
            student_vecs = uniform_feature_sampling(student_vecs, teacher_vecs.shape[0])

            student_video_feature = uniform_feature_sampling(student_vecs, self.max_ctx_len)
            student_video_feature = torch.from_numpy(l2_normalize_np_array(student_video_feature))

            teacher_video_feature = uniform_feature_sampling(teacher_vecs, self.max_ctx_len)
            # teacher_video_feature = torch.from_numpy(teacher_video_feature)
            teacher_video_feature = torch.from_numpy(l2_normalize_np_array(teacher_video_feature))

            return student_video_feature, teacher_video_feature, index, video_id
        else:
            student_video_feature = uniform_feature_sampling(student_vecs, self.max_ctx_len)
            student_video_feature = torch.from_numpy(l2_normalize_np_array(student_video_feature))
            return student_video_feature, index, video_id

    def __len__(self):
        return self.length


class TxtDataSet4DLDKD(data.Dataset):
    """
    Load captions
    """

    def __init__(self, cap_file, text_feat_path,opt):
        # Captions
        self.captions = {}
        self.cap_ids = []
   
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)

        self.text_feat_path = text_feat_path
        self.max_desc_len = opt.max_desc_l
        self.length = len(self.cap_ids)
        self.text_feat = h5py.File(self.text_feat_path, 'r')
        self.teacher_feat = None
        
        self.student = opt.student
        self.opt = opt
       

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        
        cap_feat = self.text_feat[cap_id][...]
        cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat)).squeeze()[:self.max_desc_len]
        if self.teacher_feat is not None:
            
            if self.opt.collection == 'charades':
                teacher_cap = torch.from_numpy(l2_normalize_np_array(self.teacher_feat[cap_id][...])).float()
            else:
                teacher_cap = torch.from_numpy(l2_normalize_np_array(self.teacher_feat['#'.join(cap_id.split('#enc#'))][...])).float()
            return cap_tensor, teacher_cap, index, cap_id
        else:
            return cap_tensor, index, cap_id

    def __len__(self):
        return self.length

if __name__ == '__main__':
    pass


