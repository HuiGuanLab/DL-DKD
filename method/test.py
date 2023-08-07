from clip import load,tokenize
import os
import h5py
import torch
from PIL import Image
import cv2
import numpy as np

# for i in ["ViT-B/16","ViT-B/32","ViT-L/14@336px","ViT-L/14","RN50","RN101","RN50x4","RN50x16","RN50x64"]:
clip_model='ViT-B/32'
def get_query_feat(dataset):
    model,_ = load(clip_model,device='cpu')
    file_name=f'/home/zms/VisualSearch/{dataset}/TextData/clip_ViT_B_32_{dataset}_query_feat.hdf5'
    if not os.path.exists(file_name):
        f = h5py.File(file_name, 'w')
    else:
        f = h5py.File(file_name, 'a')

    sps=['train','val']
    if dataset=='didemo':
        sps.append('test')
    for split in sps:
        s = 0
        with open(f'/home/zms/VisualSearch/{dataset}/TextData/{dataset}{split}.caption.txt','r') as reader:
            data = reader.readlines()
            for i in data:
                s+=1
                print(s)
                text_id, cap = i.split(' ',1)
                cap=cap.strip()
                cap = tokenize([cap])
                with torch.no_grad():
                    e = model.encode_text(cap)
                f.create_dataset(text_id, data=e)
    f.close()
def get_vid_feat(dataset):
    device = torch.device("cuda:0")
    model, preprocess = load(clip_model, device=device)
    file_name = f'/home/zms/VisualSearch/{dataset}/FeatureData/clip_ViT_L_14_{dataset}_vid_feat.hdf5'
    if not os.path.exists(file_name):
        f = h5py.File(file_name, 'w')
    else:
        f = h5py.File(file_name, 'a')
    vid_name = os.listdir(f'/home/zms/VisualSearch/{dataset}/VideoData')
    with open('/home/zms/VisualSearch/activitynet/video2frames.txt', 'r') as reader:  # v_xwSeXFkTNlE
        data = reader.read()
        data = eval(data)
    # flag=True
    for vid in vid_name:
        print(vid)
        # if flag:
        #     if vid !='v_Ed08LA1pjIg.mp4':
        #         continue
        #     else:
        #         flag=False
        asd=0
        i=0
        path = os.path.join(f'/home/zms/VisualSearch/{dataset}/VideoData/{vid}')
        vc = cv2.VideoCapture(path)  # 读取视频 或者获取摄像头
        vid = vid.rsplit('.',1)[0]
        print(vid)
        ret = True
        total_length=0
        while ret:
            ret, _ = vc.read()
            total_length+=1
        print('video_total_length'+str(total_length))
        frame_count = int(total_length/len(data[vid]))
        print('frame_tiao:'+str(frame_count))
        ret=True
        vc = cv2.VideoCapture(path)  # 读取视频 或者获取摄像头
        while ret:
            ret, frame = vc.read()
            if ret and asd%frame_count==0 and i<=len(data[vid]):
                frame = Image.fromarray(frame)
                frame = preprocess(frame).unsqueeze(0).to(device)
                with torch.no_grad():
                    e = model.encode_image(frame)
                    e = e.detach().cpu().numpy()
                if f'{vid}_{i}' not in f.keys():
                    f.create_dataset(f'{vid}_{i}', data=e)
                if i%50==0:
                    print(i)
                i+=1
            asd+=1
        # f.flush()

    f.close()


# with open('/home/zms/VisualSearch/activitynet/video2frames.txt','r') as reader: #v_xwSeXFkTNlE
#     data=reader.read()
#     data=eval(data)
#     print(len(data['v_jsxrJJkUl2E']))
#     exit()

# get_vid_feat('activitynet')

get_query_feat('tvr')

