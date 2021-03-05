import os
import pdb
import sys
from pathlib import Path
import h5py
import pickle
import numpy as np
import json
import glob

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose, ToTensor, Normalize

import ml
from ml import io, logging


class CATER_DataSet(Dataset):

    def __init__(self, args, num_classes, root_path, vid_feat_folder, split, num_frames):
        self.args = args
        
        num_classes = slice(*num_classes) if isinstance(num_classes, list) else slice(None, num_classes, None)
        self.root = Path(root_path)
        self.vid_feats_path = Path(os.path.join(root_path, vid_feat_folder, split))
        self.classes = sorted([d.name for d in self.vid_feats_path.iterdir() if d.is_dir()])
        self.class2idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.idx2class = {self.class2idx[key]: key for key in self.class2idx}
        
        self.classes = self.classes[num_classes]

        self.vid_feats_paths = []
        for class_label in self.classes:
            self.vid_feats_paths.extend(list(self.vid_feats_path.glob(f"{class_label}/*.h5")))
        self.vid_feats_paths.sort()
        
        self.split = split
        self.num_frames = num_frames
        
        if ('feature_ready' in self.args) and self.args.feature_ready:
            self.feature_path = os.path.join(self.args.data, self.args.dataset, self.args.feature_path)
        

    def __getitem__(self, index):
        current_vid_feats_path = self.vid_feats_paths[index]
        target = self.class2idx[current_vid_feats_path.parent.name]
        video_name = current_vid_feats_path.stem
        
        if ('feature_ready' in self.args) and self.args.feature_ready:
            h5f = io.load(os.path.join(self.feature_path, video_name + '.h5'))
            
            tracks_emb = h5f.tracks_emb
            frame_feat = h5f.frame_feat
            tracks_visibility_mask = h5f.tracks_visibility_mask
            heuris_last_visible_snitch = h5f.heuris_last_visible_snitch
            tracks_boxes = h5f.tracks_boxes
            tracks_labels = h5f.tracks_labels
            
            return (tracks_emb, frame_feat, heuris_last_visible_snitch, 
                    tracks_visibility_mask, tracks_boxes, tracks_labels,
                    target, video_name)
            
        else:
            try:
                h5f = io.load(current_vid_feats_path)
            except:
                print(current_vid_feats_path, 'h5 file open error!')
                
            frame_feat_maps = h5f.image_feat_maps 
            frame_feat = h5f.image_tensor
            h5f.close()


            if frame_feat_maps.size(0) < self.num_frames:  # a few videos have smaller frames, pad with the last frame
                last_frame_feat_maps = frame_feat_maps[-1, :, :, :].repeat(self.num_frames-frame_feat_maps.size(0), 1, 1, 1)
                frame_feat_maps = torch.cat([frame_feat_maps, last_frame_feat_maps], axis=0)

                last_frame_feat = frame_feat[-1, :].repeat(self.num_frames-frame_feat.size(0), 1)
                frame_feat = torch.cat([frame_feat, last_frame_feat], axis=0)



            if frame_feat_maps.size(0) > self.num_frames:  # if the video has more frames than it is supposed to be, omit the beginning ones
                frame_feat_maps = frame_feat_maps[len(frame_feat_maps) - self.num_frames:]
                frame_feat = frame_feat[len(frame_feat) - self.num_frames:]

            return frame_feat_maps, frame_feat, target, video_name

    
    def __len__(self):
        return len(self.vid_feats_paths)

