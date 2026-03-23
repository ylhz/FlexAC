# qwen_vl_norm_cos_cuowei_sigmoid
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import copy as cp
from .base import BaseModel
from ..smp import isimg, listinstr
from ..dataset import DATASET_TYPE
from ipdb import set_trace


def get_control_vector_topN(control_path, dis_type="euclidean", top_N=0):

    if control_path is None or control_path=='':
        print("\033[91m control_path is None, return None \033[0m ")
        return None
    # set_trace()
    if os.path.exists(os.path.join(control_path, "text_normal_fea.pt")):
        key_ans = ['text_normal', 'text_creative']
        feature_dict = {key: torch.load(os.path.join(control_path, f"{key}_fea.pt")).float() for key in key_ans}  # [32, n, 4096]
    else:
        key_ans = ['fea_n', 'fea_c']
        feature_dict = {
            "text_normal": torch.load(os.path.join(control_path, f"{key_ans[0]}.pt")).float(),
            "text_creative": torch.load(os.path.join(control_path, f"{key_ans[1]}.pt")).float(),
            }  # [32, n, 4096]

    # 记录特征距离
    fea_dis_all = []
    # 交错
    vectors = []
    layers = list(range(feature_dict['text_normal'].shape[0]))

    # 特征差异
    for l in layers:
        fea_n = feature_dict['text_normal'][l]
        fea_c = feature_dict['text_creative'][l]

        fea_n_odd, fea_n_even = fea_n[1::2, :], fea_n[0::2, :]
        fea_c_odd, fea_c_even = fea_c[1::2, :], fea_c[0::2, :]

        fea_diff = ( (fea_c_even - fea_n_odd) + (fea_c_odd - fea_n_even) ) / 2

        # 计算normal和creative特征距离
        if dis_type == "euclidean":
            # 欧氏距离
            distances1 = torch.norm(fea_c_even - fea_n_odd, dim=-1)  # [N]
            distances2 = torch.norm(fea_c_odd - fea_n_even, dim=-1)  # [N]
        elif dis_type == "cosine":
            # 余弦距离
            distances1 = 1 - torch.nn.functional.cosine_similarity(fea_c_even, fea_n_odd, dim=-1)  # [N]
            distances2 = 1 - torch.nn.functional.cosine_similarity(fea_c_odd, fea_n_even, dim=-1)  # [N]
        dis_mean = (distances1 + distances2) / 2
        fea_dis_all.append(dis_mean)

        vectors.append(fea_diff)
    delta = torch.stack(vectors, dim=0)  # [32, n, 4096]

    # 筛选,找到距离最大的前N个
    fea_dis_all = torch.stack(fea_dis_all, dim=0)
    fea_dis_all = fea_dis_all.mean(dim=0).cpu().numpy() # [N]
    top_N_index = np.argsort(-fea_dis_all)[:top_N]  # TOP
    print("TOP:", top_N_index)

    delta_topN = delta[:, top_N_index, :]  # [32, topN, 4096]
    diff = delta_topN.mean(dim=1)  # [32, 4096]

    control_vector = diff 
    return control_vector

def edit_feature_cos(ori_fea, control_vector, control_factor=1, cur_layer=0):
    """
    Args:
    ori_fea: [N,token,dim],
    control_vector: [1, dim]
    """

    def sigmoid(x):
        return 1 / (1 + torch.exp(-100 * x))
    
    control_vector = control_factor * control_vector[cur_layer].unsqueeze(0)  # [1, dim]

    # Expand control_vector to match ori_fea shape
    control_vector_expanded = control_vector.unsqueeze(1).expand(ori_fea.size(0), ori_fea.size(1), -1)  # [N, token, dim]

    # Compute cosine similarity along dim=-1
    cos_sim = - torch.nn.functional.cosine_similarity(ori_fea, control_vector_expanded, dim=-1)  # [N, token]

    # Set negative values to zero
    cos_sim = cos_sim.clamp_min(0.0).unsqueeze(-1)  # OR: cos_sim[cos_sim < 0] = 0
    # print("cos_sim:", *cos_sim.cpu().to(torch.float32).numpy())
    cos_sim = sigmoid(cos_sim) 
    # set_trace()

    new_fea = ori_fea + cos_sim  * control_vector_expanded

    norm_fea_old = torch.norm(ori_fea, p=2, dim=-1, keepdim=True)  # shape: [N, m, 1]
    norm_fea = torch.norm(new_fea, p=2, dim=-1, keepdim=True).clamp(min=1e-8)  # shape: [N, m, 1]
    
    # 缩放
    new_fea = new_fea * (norm_fea_old / norm_fea)

    return new_fea



class QwenVLChatFlexAC_TOP(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='Qwen/Qwen-VL-Chat', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cuda', trust_remote_code=True).eval()
        torch.cuda.empty_cache()
        self.kwargs = kwargs
        print("QwenVL-CHAT.")

        # for steering vector
        use_sample_ = os.environ.get('is_use_sample', None)
        use_sample = use_sample_ == str(True)
        if use_sample:
            self.kwargs['do_sample'] = True
            self.kwargs['temperature'] = 1.0
        else:
            self.kwargs['do_sample'] = False


        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')
        
        self.output_type = "open_end"
        self.layer_len = len(self.model.transformer.h)
        # self.control_layers = [15, 16, 17]

        # 层位置
        control_layers = os.environ.get('CONTROL_LAYERS', "15 16 17")
        self.control_layers =  list(map(int,control_layers.split(' '))) 
        print("control_layers: ", self.control_layers)


        self.layer_idxs = list(range(self.layer_len))
        self.cur_layer_idx = -1
        control_path = os.environ.get('RUN_NAME_EVAL_CREATION_FLEXAC_BENCH_PATH', None)
        self.control_factor = float(os.environ.get('FLEXAC_CONTROL_FACTOR', 0))
        self.control_topN = int(os.environ.get('FLEXAC_TOPN', 0))
        self.distance_type = os.environ.get('FLEXAC_DISTANCE_TYPE', 0)
        self.control_vector = get_control_vector_topN(control_path=control_path, top_N=self.control_topN, dis_type=self.distance_type)
        if self.control_vector is not None:
            # 在每层最后一个block的输出上添加hook
            for layer_idx in self.layer_idxs:
                self.add_control_hook(self.model.transformer.h[layer_idx])


    def add_control_hook(self, module):
        
        def hook(module, input, output):
            self.cur_layer_idx = (self.cur_layer_idx + 1) % len(self.layer_idxs)

            cur_layer = self.layer_idxs[self.cur_layer_idx]

            # open_end模式下，所有token的特征都要调整
            if self.output_type == "open_end":
                if cur_layer in self.control_layers:
                    output_list = list(output)  # 将元组转换为列表
                    out_tensor = output_list[0][:,-2:,:]

                    output_list[0][:,-2:,:] = edit_feature_cos(out_tensor, self.control_vector.to(out_tensor.device), self.control_factor, cur_layer)
                    output = tuple(output_list)  # 将列表转换回元组

                return output

        module.register_forward_hook(hook)
    

    def build_history(self, message):

        def concat_tilist(tilist):
            image_cnt = 1
            prompt = ''
            for item in tilist:
                if item['type'] == 'text':
                    prompt += item['value']
                elif item['type'] == 'image':
                    prompt += f"Picture {image_cnt}: <img>{item['value']}</img>\n"
                    image_cnt += 1
            return prompt

        assert len(message) % 2 == 0
        hist = []
        for i in range(len(message) // 2):
            m1, m2 = message[2 * i], message[2 * i + 1]
            assert m1['role'] == 'user' and m2['role'] == 'assistant'
            hist.append((concat_tilist(m1['content']), concat_tilist(m2['content'])))
        return hist

    def generate_inner(self, message, dataset=None):
        vl_list = [{'image': s['value']} if s['type'] == 'image' else {'text': s['value']} for s in message]
        query = self.tokenizer.from_list_format(vl_list)
        response, _ = self.model.chat(self.tokenizer, query=query, history=None, **self.kwargs)
        return response

    def chat_inner(self, message, dataset=None):
        assert len(message) % 2 == 1 and message[-1]['role'] == 'user'
        history = self.build_history(message[:-1])
        vl_list = [
            {'image': s['value']} if s['type'] == 'image' else {'text': s['value']}
            for s in message[-1]['content']
        ]
        query = self.tokenizer.from_list_format(vl_list)
        response, _ = self.model.chat(self.tokenizer, query=query, history=history, **self.kwargs)
        return response
