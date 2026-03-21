# 1. 系统
# 2. 正则
# 3. json
# 4. 字符串
import os
import re
import json
import string

# 5. prompt模版
# 6. numpy数组
# 7. 基本类型
# 8. 计数器
from prompt_template import get_prompt
import numpy as np
from typing import List, Union
from collections import Counter

# 9. torch
# 10. transformers库中的tokenizer，model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 11. 项目路径，从而获取Data Augmentation路径
from root_dir_path import ROOT_DIR
DATA_ROOT_DIR = os.path.join(ROOT_DIR, "data_aug")


# 1. 获取模型
# 1.1 根据model_name，返回model, tokenizer, 推理配置
def get_model(model_name, max_new_tokens=20):
    # 1. 获取模型
    # 1.1 根据model_name，获取模型HuggingFace仓库名
    # 1.2 下载模型
    # 1.3 获取tokenizer
    model_path = get_model_path(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, low_cpu_mem_usage=False, device_map=None, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 2. 设置推理配置
    # 2.1 Greedy Decoding策略，每次一样的prompt可以生成一样的回答；因为num_beams = 1，do_sample=False，也就是只保留1个候选token，也不随机采样，所以是确定性生成
    # 2.2 最多Decoding出20个新token
    # 2.3 返回除了生成的token外，还有attention，score等更多信息
    # 2.4 pad token id设定
    generation_config = dict(
        num_beams=1, do_sample=False,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
    )

    # 3. 返回
    # 3.1 返回模型，tokenizer，推理配置
    return model, tokenizer, generation_config


# 1.2 获取模型HuggingFace仓库名
def get_model_path(model_name):
    if model_name == "llama3-8b-instruct": 
        return "meta-llama/Meta-Llama-3-8B-Instruct"
    elif model_name == "qwen2.5-1.5b-instruct":
        return "Qwen/Qwen2.5-1.5B-Instruct"
    elif model_name == "llama3.2-1b-instruct":
        return "meta-llama/Llama-3.2-1B-Instruct"
    else:
        return model_name

# 2. 加载数据
# 2.1 根据数据集名称，进行数据增强时用的模型名称，返回该数据集在该模型下增强后的数据集
# 2.2 如果指定了具体的问题类型，就只返回增前后的该类型的问题
def load_data(data_name, data_type, model_name):
    # 1.1 获取数据集下json文件
    # 1.1.1 比如/PRAG/data_aug/2wikimultihopqa/llama3-8b-instruct/，有具体的5个.json文件，bridge_comparison.son, comparison.json, compositional.json, inference.json, total.json
    # 1.1.1 比如/PRAG/data_aug/<data_name>/<model_name>/，有具体的5个.json文件
    # 1.1.2 然后获取这5个json文件
    input_dir = os.path.join(DATA_ROOT_DIR, data_name, model_name)
    files = [f for f in os.listdir(input_dir)]

    # 1.2 如果有2个及以上文件
    solve_dataset = []
    if len(files) > 1: 
        # 1.2.1 指定获取total类型
        # 1.2.1.1 那就读取所有json文件，合并数据，然后返回
        if data_type == "total": 
            all_data = {}
            for filename in files:
                with open(os.path.join(input_dir, filename), "r") as fin:
                    all_data[filename] = json.load(fin)
            total_data = []
            idx = {filename: 0 for filename in files}
            for data in all_data["total.json"]:
                typ = data["type"] + ".json"
                if idx[typ] == len(all_data[typ]):
                    break 
                aim_data = all_data[typ][idx[typ]]
                assert aim_data["question"] == data["question"]
                idx[typ] += 1
                total_data.append(aim_data)
            return [["total.json", total_data]]
        
        # 1.2.2 指定获取具体问题类型
        for filename in files:
            if filename != "total.json":
                with open(os.path.join(input_dir, filename), "r") as fin:
                    solve_dataset.append((filename, json.load(fin)))
        if data_type is None:
            return solve_dataset
        else:
            data_type = data_type + ".json"
            if data_type not in [v[0] for v in solve_dataset]:
                raise ValueError(f"Invalid {data_type} in Dataset {data_name}")
            tmp = []
            for filename, dataset in solve_dataset:
                if filename == data_type:
                    tmp.append((filename, dataset))
            return tmp
    
    # 1.3 如果只有1个文件
    # 1.3.1 那这个文件就是total.json，直接读取返回
    else:
        with open(os.path.join(input_dir, "total.json"), "r") as fin:
            solve_dataset.append(("total.json", json.load(fin)))
        return solve_dataset
 

####################################################################################################
class BaseDataset:
    @classmethod
    def normalize_answer(cls, s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def exact_match_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))

        correct = np.max([int(cls.normalize_answer(prediction) == cls.normalize_answer(gt)) for gt in ground_truths])
        return {'correct': correct, 'incorrect': 1 - correct}

    @classmethod
    def f1_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))
            
        final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
        for ground_truth in ground_truths:
            normalized_prediction = cls.normalize_answer(prediction)
            normalized_ground_truth = cls.normalize_answer(ground_truth)
            if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ['f1', 'precision', 'recall']:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric

# -------------------------------- for augmentation ----------------------------------------

def model_generate(prompt, model, tokenizer, generation_config):
    messages = [{
        'role': 'user', 
        'content': prompt,
    }]
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True
    )
    input_len = len(input_ids)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    output = model.generate(
        input_ids, 
        attention_mask = torch.ones(input_ids.shape).to(model.device),
        **generation_config
    )
    output = output.sequences[0][input_len:]
    text = tokenizer.decode(output, skip_special_tokens=True)
    return text

# ------------------------------------------------------------------------------------

def read_complete(filepath):
    try:
        with open(filepath, "r") as fin:
            data = json.load(fin)
        return data, len(data)
    except:
        return [], 0

    
def evaluate(pred, ground_truth, with_cot=False):
    if not with_cot:
        pred = pred.strip()
        stop_list = [".", "\n", ","]
        for stop in stop_list:
            end_pos = pred.find(stop)
            if end_pos != -1:
                pred = pred[:end_pos].strip()
    else:
        if "the answer is" in pred:
            pred = pred[pred.find("the answer is") + len("the answer is"):]
        pred = pred.strip()
        stop_list = [".", "\n", ","]
        for stop in stop_list:
            end_pos = pred.find(stop)
            if end_pos != -1:
                pred = pred[:end_pos].strip() 

    em = BaseDataset.exact_match_score(
        prediction=pred,
        ground_truth=ground_truth,
    )["correct"]
    f1_score = BaseDataset.f1_score(
        prediction=pred,
        ground_truth=ground_truth,
    )
    f1, prec, recall = f1_score["f1"], f1_score["precision"], f1_score["recall"]
    return {
        "eval_predict": pred,
        "em": str(em),
        "f1": str(f1),
        "prec": str(prec),
        "recall": str(recall),
    }


def predict(model, tokenizer, generation_config, question, with_cot, passages = None):
    model.eval()
    input_ids = get_prompt(
        tokenizer, 
        question, 
        passages = passages, 
        with_cot = with_cot)
    input_len = len(input_ids)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids, 
            attention_mask = torch.ones(input_ids.shape).to(model.device),
            **generation_config)
    output = output.sequences[0][input_len:]
    text = tokenizer.decode(output, skip_special_tokens=True)
    return text