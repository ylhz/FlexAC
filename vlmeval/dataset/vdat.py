import json
import math
import os
import sys
import numpy as np
import torch
import torchvision
from torch import nn, Tensor
from abc import abstractmethod
from typing import List, Any, Callable

from tqdm import tqdm

# from torchtext.vocab import GloVe

import open_clip
from torchvision import transforms
from PIL import Image
import pandas as pd

import nltk
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag

from transformers import AutoTokenizer, AutoModel

from sentence_transformers import SentenceTransformer, util  # pip install -U sentence-transformers==2.0.6

# from concurrent.futures import ThreadPoolExecutor
# from multiprocessing import Pool, cpu_count
import multiprocessing as mp
# import concurrent.futures
import re

def extract_noun(input_string):
    if not isinstance(input_string, str):
        raise TypeError(f"Expected string or bytes-like object, but got {type(input_string)}")
    
    # 1. Umbrella\n2. Hat\n3. Graduation\n4. Cap\n5. Gown\n6. Brick\n7. Flower\n8. Paper\n9. Building\n10. Hat
    # "Train, platform, blue, yellow, black, white, silver, light, pole, station, night."

    # # 匹配数字点号后跟随的单词
    # pattern = re.compile(r'\d+\.\s*([A-Za-z]+)')
    # # 匹配逗号分隔的单词
    # pattern2 = re.compile(r'\b([A-Za-z]+)\b')

    # 匹配数字点号后跟随的单词或短语
    pattern = re.compile(r'\d+\.\s*([A-Za-z\s]+)')
    # 匹配逗号分隔的单词或短语
    pattern2 = re.compile(r'\b([A-Za-z]+(?:\s+[A-Za-z]+)*)\b')

    # 找到所有匹配的单词
    # matches1 = pattern.findall(input_string)
    matches1 = [match.strip() for match in pattern.findall(input_string)]
    # matches2 = pattern2.findall(input_string.split('\n')[-1])
    matches2 = [match.strip() for match in pattern2.findall(input_string.split('\n')[-1])]

    if len(matches1) >= 10:
        return matches1[:10]
    elif len(matches2) >= 10:
        return matches2[:10]
    elif len(matches1) > 0 or len(matches2) > 0:
        # print("=====================================")
        # print("input_string: ", input_string)
        # print(f"matches1: {matches1}, matches2: {matches2}")
        return []
        # return matches1 + matches2
    else:
        return []






def get_upper_triangular_elements(matrix):
    # 获取上三角部分（不包含对角线）
    assert matrix.shape[0] == matrix.shape[1]
    upper_triangular_indices = np.triu_indices_from(matrix, k=1)
    upper_triangular_elements = matrix[upper_triangular_indices]
    return upper_triangular_elements


class BaseFeatureExtractor(nn.Module):
    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

class OpenClipFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, model_name='RN50-quickgelu', pretrained_date=None, device='cuda'):
        super(OpenClipFeatureExtractor, self).__init__()
        # self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # self.model = open_clip.create_model_and_transforms(model_name, pretrained='cc12m', device=device)

        openclip, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_date, device=device)
        self.model = openclip
        # self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        

        if model_name.endswith('336'):
            image_size = 336
        else:
            image_size = 224

        self.normalizer = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.Normalize(mean=self.model.visual.image_mean, std=self.model.visual.image_std),
            ]
        )
        self.encode_text = self.model.encode_text
        self.model_name = model_name
    
    def forward(self, x):
        x = torch.clamp(x, min=0, max=1)
        # inputs = self.processor(images=x, return_tensors="pt")
        inputs = self.normalizer(x)
        outputs = self.model.encode_image(inputs)
        pooled_output = outputs
        # print(f"Clip {pooled_output.shape}")
        return pooled_output
    


###### CLIP similarity ######
class CLIPSimilarity:
    def __init__(self, model_name='RN50-quickgelu', pretrained_date=None, prompt="A photo of "):
        self.model = OpenClipFeatureExtractor(model_name, pretrained_date)
        self.model = self.model.eval().requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.prompt = prompt

    def get_text_features(self, texts):
        # model = self.model.eval().requires_grad_(False)
        text_features = []
        with torch.no_grad():
            for text in texts:
                text = self.prompt + text + "."

                text_feature = self.model.encode_text(self.tokenizer(text).cuda())
                text_feature = torch.nn.functional.normalize(text_feature, dim=-1).mean(dim=0)
                text_feature /= text_feature.norm()

                text_features.append(text_feature)
        if len(text_feature.shape) == 1:
            text_features = torch.stack(text_features, dim=0)
        else:
            text_features = torch.cat(text_features, dim=0)

        return text_features

    def get_image_features(self, image):
        with torch.no_grad():
            image_features = self.model(image)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
        return image_features

    def get_similarity(self, image, texts, image_caption=None):
        assert len(texts) == 1

        if len(texts[0]) == 0:
            return None, None, None, None, None
        
        image_features = self.get_image_features(image)
        text_features = self.get_text_features(texts[0])

        # cosine similarity - image and text
        cos_img_text = (image_features @ text_features.t()).cpu().numpy()
        cos_img_text_mean = cos_img_text.mean()

        # cosine similarity - text and text
        cos_text_text_ = (text_features @ text_features.t()).cpu().numpy()
        cos_text_text = get_upper_triangular_elements(cos_text_text_)
        cos_text_text_mean = cos_text_text.mean()

        cos_mean = (cos_img_text_mean + cos_text_text_mean) / 2

        return cos_img_text, cos_img_text_mean, cos_text_text, cos_text_text_mean, cos_mean

class SemanticTextSimilarity:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2',response_num=10):
        # nltk.download('wordnet')
        # nltk.download('omw-1.4')
        # nltk.download('averaged_perceptron_tagger')
        # nltk.download('punkt')
        # wn.ensure_loaded() 
        self.model = SentenceTransformer(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.invalid_num = 0
        self.response_num = response_num

    def semantic_text_similarity(self, text1, text2):
        embeddings1 = self.model.encode(text1, convert_to_tensor=True)
        embeddings2 = self.model.encode(text2, convert_to_tensor=True)
        cos_sim = util.pytorch_cos_sim(embeddings1, embeddings2)
        return cos_sim.item()

    def compute_similarity(self, texts, i, j):
        similarity = self.semantic_text_similarity(texts[i], texts[j])
        return (i, j, similarity)
    
    def get_text_similarity(self, texts):
        # texts: ["Apple", "Phone", "Camera", "Sunglasses", "Lens", "Reflection", "Shadow", "Light", "Glass", "Refraction"]
        n = len(texts)
        # cos_text_text = np.zeros((n, n))
        cos_text_text = []

        # 并行计算
        # with ThreadPoolExecutor() as executor:
        #     futures = [
        #         executor.submit(self.compute_similarity, texts, i, j)
        #         for i in range(n) for j in range(i + 1, n)
        #     ]

        #     for future in futures:
        #         i, j, similarity = future.result()
        #         cos_text_text[i][j] = similarity
        #         cos_text_text[j][i] = similarity

        # 串行计算
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                cos_text_text.append(self.semantic_text_similarity(texts[i], texts[j]))

        return cos_text_text

    def extract_noun_from_caption(self, caption):
        # 1. 提取描述中的所有单词
        tokens = word_tokenize(caption)
        # 2. 进行词性标注
        tagged_tokens = pos_tag(tokens)
        # 3. 将所有单词转换为名词
        nouns = []

        for word, tag in tagged_tokens:
            if tag.startswith('NN'):  # 名词
                nouns.append(word)
            elif tag in ['JJ', 'VB', 'RB']:  # 形容词、动词、副词
                synsets = wn.synsets(word, pos=wn.NOUN)
                if synsets:
                    nouns.append(word)

        # 4. 去掉专有名词和代名词
        filtered_nouns = []
        for word, tag in pos_tag(nouns):
            if tag not in ['NNP', 'NNPS', 'PRP', 'PRP$']:  # 专有名词和代名词
                filtered_nouns.append(word)
        return filtered_nouns

    def compute_max_similarity(self, noun, img_nouns):
        similarities = [self.semantic_text_similarity(noun, word) for word in img_nouns]
        try:
            return max(similarities)
        except:
            print(f"noun: {noun}, img_nouns: {img_nouns}")
            input()
    
    def get_image_text_similarity(self, image_caption, texts, img_nouns):
        # img_nouns = self.extract_noun_from_caption(image_caption)

        # TYPE I: 并行计算
        # with ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(self.compute_max_similarity, noun, img_nouns) for noun in texts]
        #     cos_img_text = np.array([future.result() for future in futures])

        # TYPE II: 串行计算
        cos_img_text = []
        for i, noun in enumerate(texts):
            cos_img_text.append(self.compute_max_similarity(noun, img_nouns))
            # similarities = [self.semantic_text_similarity(noun, word) for word in img_nouns]
            # cos_img_text[i] = max(similarities)

        if len(cos_img_text) == 0:
            return None, None
        
        cos_img_text_mean = sum(cos_img_text) / len(cos_img_text)
        return cos_img_text, cos_img_text_mean
    
    def outer_similarity(self, multi_texts):
        assert isinstance(multi_texts[0], list)
        cos_text_text = []
        for i in range(len(multi_texts)):
            for j in range(i+1, len(multi_texts)):
                ans1 = ", ".join(multi_texts[i])
                ans2 = ", ".join(multi_texts[j])
                cos_ = self.semantic_text_similarity(ans1, ans2)
                cos_text_text.append(cos_)
        
        if len(cos_text_text) == 0:
            return None
        cos_outer = sum(cos_text_text) / len(cos_text_text)
        return cos_outer

    def process_texts(self, multi_agrs):
        image_caption, texts, img_nouns = multi_agrs
        # cosine similarity - image and text
        cos_img_text, cos_img_text_mean = self.get_image_text_similarity(image_caption, texts, img_nouns)
        
        # cosine similarity - text and text
        cos_text_text_ = self.get_text_similarity(texts)
        # cos_text_text = get_upper_triangular_elements(cos_text_text_)
        if len(cos_text_text_) == 0:
            return None, None
        cos_text_text_mean = sum(cos_text_text_)/len(cos_text_text_)

        torch.cuda.empty_cache()

        return cos_img_text_mean, cos_text_text_mean

    def get_similarity(self, image, multi_texts, image_caption=None):
        # import pdb; pdb.set_trace()
        # 目前只支持多个答案
        if type(multi_texts[0]) == str:
            multi_texts = [multi_texts]
        assert len(multi_texts) == self.response_num, f"{len(multi_texts)}!= {self.response_num}"
        assert isinstance(multi_texts[0], list)
        # calculate the invaild number
        for texts in multi_texts:
            if len(texts) == 0:
                self.invalid_num += 1

        wn.ensure_loaded()
        if image_caption is not None:
            img_nouns = self.extract_noun_from_caption(image_caption)
            if len(img_nouns) == 0:
                image_caption = image_caption.lower()
                img_nouns = self.extract_noun_from_caption(image_caption)
            if len(img_nouns) == 0:
                print(f"image_caption: {image_caption}")
                input()
            assert len(img_nouns) > 0
        else:
            img_nouns = None

        if len(multi_texts) == 1:
            texts = multi_texts[0]
            if len(texts) == 0:
                return None, None, None, None, None
            # cosine similarity - image and text
            cos_img_text_mean, cos_text_text_mean = self.process_texts((image_caption, texts, img_nouns))
            cos_img_text_means, cos_text_text_means = [cos_img_text_mean], [cos_text_text_mean]
        else:
            # 并行运算
            ctx = mp.get_context('spawn')
            with ctx.Pool(10) as pool:
                results = pool.map(self.process_texts, [(image_caption, texts, img_nouns) for texts in multi_texts])
                
            cos_img_text_means, cos_text_text_means = zip(*results)
        
        # 串行运算
        # cos_img_text_means, cos_text_text_means = [], []
        # for i in range(len(multi_texts)):
        #     texts = multi_texts[i]

        #     # cosine similarity - image and text
        #     cos_img_text, cos_img_text_mean = self.get_image_text_similarity(image_caption, texts, img_nouns)
            
        #     # cosine similarity - text and text
        #     cos_text_text_ = self.get_text_similarity(texts)
        #     cos_text_text = get_upper_triangular_elements(cos_text_text_)
        #     cos_text_text_mean = cos_text_text.mean()

        #     cos_img_text_means.append(cos_img_text_mean)
        #     cos_text_text_means.append(cos_text_text_mean)

        #     cos_mean = (cos_img_text_mean + cos_text_text_mean) / 2

        # remove None in list
        cos_img_text_means = [value for value in cos_img_text_means if value is not None]
        cos_text_text_means = [value for value in cos_text_text_means if value is not None]

        if len(cos_img_text_means) == 0 or len(cos_text_text_means) == 0:
            print(f"cos_img_text_means: {cos_img_text_means}, cos_text_text_means: {cos_text_text_means}")
            import pdb; pdb.set_trace()

        # inner similarity
        cos_img_text_means = np.array(cos_img_text_means)
        cos_text_text_means = np.array(cos_text_text_means)

        # outer similarity
        cos_outer = self.outer_similarity(multi_texts)
        
        return cos_img_text_means, cos_img_text_means.mean(), cos_text_text_means, cos_text_text_means.mean(), cos_outer
