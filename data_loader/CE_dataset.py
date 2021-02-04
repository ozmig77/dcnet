import os
import json
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from pathlib import Path
from utils import memcache, read_json
from PIL import Image

class CE(Dataset):

    def __init__(self, data_dir, text_dim, category, expert_dims, split, text_feat,
                 max_text_words, max_expert_tokens, transforms):
        self.ordered_experts = list(expert_dims.keys())
        self.transforms = transforms
        self.text_dim = text_dim
        self.max_text_words = max_text_words
        self.max_expert_tokens = max_expert_tokens
        self.category = category
        self.image_dir = os.path.join(data_dir, "resized_images")
        self.split = split
        self.no_target = (split == 'val_trg' or split == 'test_trg')
        self.text_feat = text_feat
        if self.no_target:
            self.data = read_json(Path(data_dir + f"/image_splits/split.{category}.{split.split('_')[0]}.json"))
            self.data_length = len(self.data)
        else:
            if text_feat == "w2v":
                text_path = f"/captions/cap.{category}.w2v.{split}.pkl"
                self.data = memcache(data_dir + text_path)
            elif text_feat == "glove":
                text_path = f"/captions/cap.{category}.glove.{split}.pkl"
                self.data = memcache(data_dir + text_path)
            else:
                raise ValueError(f"Text features {text_feat} not recognized")

            self.data_length = len(self.data)
            
    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        if self.no_target:
            candidate_id = self.data[index]
            target_id = None
            captions = None
            text = None
        elif self.split == 'test':
            candidate_id = self.data[index]['candidate']
            target_id = None
            captions = self.data[index]['captions']
            if self.text_feat in ['w2v', 'glove']:
                text = self.data[index]['wv']
            else:
                raise ValueError
        else:
            candidate_id = self.data[index]['candidate']
            target_id = self.data[index]['target']
            captions = self.data[index]['captions']

            if self.text_feat in ['w2v', 'glove']:
                text = self.data[index]['wv']
            else:
                raise ValueError
        # index
        candidate_experts = {}
        target_experts = {}
        # im_feat
        if 'im_feat' in self.ordered_experts:
            candidate_fname = candidate_id + '.jpg'
            candidate_image_orig = Image.open(os.path.join(self.image_dir, candidate_fname)).convert('RGB')
            candidate_image = self.transforms(candidate_image_orig)
            candidate_experts['im_feat'] = candidate_image
            if target_id is not None:
                target_fname = target_id + '.jpg'
                target_image_orig = Image.open(os.path.join(self.image_dir, target_fname)).convert('RGB')
                target_image = self.transforms(target_image_orig)
                target_experts['im_feat'] = target_image

        meta_info = {'candidate': candidate_id, 'target': target_id, 'captions': captions}

        return text, candidate_experts, target_experts, meta_info

    @staticmethod
    def crop_image(image, keypoint, m=20):
        box = np.zeros((3, m * 2, m * 2))
        x1 = max(0, keypoint[0] - m)
        x2 = min(244, keypoint[0] + m)
        y1 = max(0, keypoint[1] - m)
        y2 = min(244, keypoint[1] + m)
        cropped_image = image[:, x1:x2, y1:y2]
        x = cropped_image.size(1)
        y = cropped_image.size(2)
        box[:, :x, :y] = cropped_image
        return box

    def collate_fn(self, batch):
        text_feature, candidate, target, meta_info = zip(*batch)
        candidate_experts, target_experts = {}, {}
        # text
        if self.no_target:
            text = None
            text_lengths = None
        else:
            batch_size = len(text_feature)
            text_lengths = [len(x) for x in text_feature]
            max_text_words = min(max(text_lengths), self.max_text_words)

            if self.text_feat in ['w2v', 'glove']:
                text = np.zeros((batch_size, max_text_words, self.text_dim))
                for i, feature in enumerate(text_feature):
                    text[i, :text_lengths[i], :] = feature[:max_text_words]
                text = torch.FloatTensor(text)
            else:
                raise ValueError

        # experts
        for expert in candidate[0]:
            if expert == 'im_feat':
                candidate_val = np.vstack([np.expand_dims(x[expert], 0) for x in candidate])
            else:
                candidate_val = np.vstack([x[expert] for x in candidate])
            candidate_experts[expert] = torch.from_numpy(candidate_val)

        if not self.no_target and not self.split == 'test':
            for expert in target[0]:
                if expert  == 'im_feat':
                    target_val = np.vstack([np.expand_dims(x[expert], 0) for x in target])
                else:
                    target_val = np.vstack([x[expert] for x in target])

                target_experts[expert] = torch.from_numpy(target_val)

        return {'text': text,
                'text_lengths': text_lengths,
                'candidate_experts': candidate_experts,
                'target_experts': target_experts,
                'meta_info': meta_info}
