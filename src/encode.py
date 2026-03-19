# 1. 系统
# 2. 垃圾回收
# 3. 时间
import os
import gc
import time

# 4. 参数解析
# 5. prompt模版
# 6. 进度显示
# 7. 随机库
# 8. numpy数组
# 9. 基本类型
import argparse
import prompt_template
from tqdm import tqdm
import random
import numpy as np
from typing import Dict, List

# 10. torch
# 11. torch中数据集类
# 12. transformers中数据集排序器
# 13. peft参数微调相关的库
import torch
from torch.utils.data import Dataset
from transformers import DefaultDataCollator
from peft import TaskType, get_peft_model, LoraConfig, PeftModel

# 14. 项目路径
# 15. 自定义工具，获取模型，加载数据
from root_dir_path import ROOT_DIR
from utils import get_model, load_data

# 1. 设置随机数
seed = 42 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 1. 程序入口，文档参数化
def main(args):
    # 1.1 加载Data Augmentation数据
    # 1.1.1 得到一个list[]，每个元素是一个样例，包含3个文档，9个问答对
    data_list = load_data(args.dataset, args.data_type, args.augment_model)
    
    # 1.2 获取模型，tokenizer，推理配置
    model, tokenizer, _generation_config = get_model(args.model_name)
    
    # 1.3 生成思维链
    # 1.3.1 默认不生成
    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)

    # 1.4 目录 + 生成随机LoRA张量
    # 1.4.1 目录 = /PARG/offline/<使用模型名>/<rank设置_alpha设置>/<随机参数>/
    # 1.4.2 随机张量 = /PARG/offline/<使用模型名>/<rank设置_alpha设置>/<随机参数>/adapter_model.safetensors
    # 1.4.2 随机张量，对于DeepSeek-V3.1-Terminus，假设62层都是FFN，每层FFN有gate，up，down3个张量，每个张量都需要A, B
    # 1.4.2 随机张量，对于DeepSeek-V3.1-Terminus，假设62层都是FFN，每层需要6个LoRA张量，一共需要372个LoRA张量
    init_adapter_path = os.path.join(ROOT_DIR, "offline", args.model_name, f"rank={args.lora_rank}_alpha={args.lora_alpha}", "base_weight")
    if not os.path.exists(os.path.join(init_adapter_path, "adapter_model.safetensors")):
        # 1.4.1 创建/DeepSeek-V3.1-Terminus/秩为10,factor为10/adapter_model.safetensors，LoRA张量，一共62*6 = 372个张量
        print("No LoRA base weight, creating...")

        # 1.4.2 LoRA微调配置
        # 1.4.2.1 设置任务是LLM，微调张量是gate, up, down，微调新增张量秩是r + factor是alpha
        # 1.4.2.2 不采用正则化，还有一个看不懂
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=['gate_proj', 'up_proj', 'down_proj'], r=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=0, inference_mode=False,
        )

        # 1.4.3 修改模型结构
        # 1.4.3.1 就是在原来模型的基础上，在62层，每一层FFN的gate, up, down这三个张量，每一个都加上(alpha / rank) * (A * B)
        model = get_peft_model(model, peft_config)
        
        # 1.4.4 模型并行
        model.is_parallelizable = True
        model.model_parallel = True
        
        # 1.4.5 生成随机LoRA参数
        # 1.4.5.1 peft会在/PARG/offline/<使用模型名>/<rank设置_alpha设置>/<随机参数>/，下生成adapter_model.safetensors文件
        # 1.4.5.2 文件内容是62*6 = 372个张量，每个shape分别是[7168, 10] + [10, 18432]，随机
        print(f'Save LoRA base weight to {init_adapter_path}')
        os.makedirs(init_adapter_path, exist_ok=True)
        model.save_pretrained(init_adapter_path)
        time.sleep(2)
        assert os.path.exists(os.path.join(init_adapter_path, "adapter_model.safetensors")) 

    
    cot_name = "cot" if args.with_cot else "direct"
    for filename, fulldata in data_list:
        filename = filename.split('.')[0] 
        print(f"### Solving {filename} ###")
        output_dir = os.path.join(
            ROOT_DIR, 
            "offline", 
            args.model_name, 
            f"rank={args.lora_rank}_alpha={args.lora_alpha}",
            args.dataset,
            f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
            f"aug_model={args.augment_model}",
            filename,
        )
        os.makedirs(output_dir, exist_ok=True)
        fulldata = fulldata if args.sample == -1 else fulldata[:args.sample]
        for did, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            augment = data["augment"]
            for pid in range(len(augment)):
                save_path = os.path.join(output_dir, f"data_{did}", f"passage_{pid}")
                if os.path.exists(os.path.join(save_path, "adapter_model.safetensors")):
                    continue
                model = train(data["question"], [augment[pid]], args, model, tokenizer, 
                            init_adapter_path, save_path)


class TrainingData(Dataset):
    ignored_id = -100

    def __init__(self, prompt_ids, tokenizer, max_length=3000):
        self.max_length = max_length
        self.dataset = []
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        for input_ids in prompt_ids:
            labels = input_ids.copy()
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
            attention_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))
            input_ids += [pad_token_id] * (max_length - len(input_ids))
            labels += [self.ignored_id] * (max_length - len(labels))
            self.dataset.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            })
        self.total_len = len(self.dataset)
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx) -> Dict[str, list]:
        return self.dataset[idx]


class TrainingDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer, device):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device
    
    def __call__(self, examples: List[Dict[str, list]]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask = tuple(
            map(lambda x: [example[x] for example in examples], ["input_ids", "labels", "attention_mask"])
        )
        return {
            "input_ids": torch.tensor(input_ids).to(self.device),
            "labels": torch.tensor(labels).to(self.device),
            "attention_mask": torch.tensor(attention_mask).to(self.device),
        }
    

def get_train_data(aug_model, augments, tokenizer, args):
    from prompt_template import get_prompt
    prompt_ids = []
    for aug in augments:
        psg = aug["passage"]
        rew = aug[f"{aug_model}_rewrite"]
        qas = aug[f"{aug_model}_qa"]
        qpa_cnt = (len(qas) + 1) // 2
        for qid, qa in enumerate(qas):
            if qid < qpa_cnt:
                for ppp in [psg, rew]:
                    prompt_ids.append(get_prompt(tokenizer, qa["question"], 
                                                    [ppp], 
                                                    qa["answer"] if not args.with_cot else qa["full_answer"], 
                                                    with_cot=args.with_cot))
            else:
                prompt_ids.append(get_prompt(tokenizer, qa["question"], 
                                                None, 
                                                qa["answer"] if not args.with_cot else qa["full_answer"], 
                                                with_cot=args.with_cot))
    return prompt_ids


def train(question, augments, args, model, tokenizer, 
          init_adapter_path, save_path):
    prompt_ids = get_train_data(args.augment_model, augments, tokenizer, args)
    train_data = TrainingData(prompt_ids, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.per_device_train_batch_size,
        collate_fn=TrainingDataCollator(tokenizer, model.device),
        shuffle=False,
    )
    model = PeftModel.from_pretrained(model, init_adapter_path, is_trainable=True)
    model.is_parallelizable = True
    model.model_parallel = True
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(model_parameters, lr=args.learning_rate)
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    model = model.unload()
    torch.cuda.empty_cache()
    gc.collect()
    return model
          

# 1. Data Parametering
if __name__ == "__main__":
    # 1. 参数
    # 1.1 必须指定训练model_name, 需要被参数化的dataset、
    # 1.2 问题类型data_type默认为total，默认不生成思考链，默认也用一样模型去增强的增强文档
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",     type=str, required=True)
    parser.add_argument("--dataset",        type=str, required=True)
    parser.add_argument("--data_type",      type=str)
    parser.add_argument("--with_cot",       action="store_true")
    parser.add_argument("--augment_model",  type=str, default=None)

    # 2. 训练，看不懂
    parser.add_argument("--per_device_train_batch_size",    type=int,   default=1)
    parser.add_argument("--num_train_epochs",               type=int,   default=3)
    parser.add_argument("--learning_rate",                  type=float, default=3e-4)

    # 3. LoRA矩阵的shape
    # 3.1 lora_rank就是秩r，也就是gate.shape = [7168, 18432]，那么A.shape = [7168, 10], B.shape = [10, 18432]，这个10就是秩r
    # 3.2 lora_alpha是scaling factor，就是gate' = gate + (系数) * A * B，这个系数是lora_alpha / lora_rank
    # 3.2.1 也就是如果lora_alpha = lora_rank的话，就是W' = W + A*B^T，如果lora_alpha = 2 * lora_rank的话，就是W' = W + 2 * A*B^T
    parser.add_argument("--lora_rank",      type=int, default=None)
    parser.add_argument("--lora_alpha",     type=int, default=None)

    # 4. 不知道干嘛的
    parser.add_argument("--sample",         type=int, default=-1) # -1 means all

    # 5. 参数检查
    args = parser.parse_args()
    assert args.lora_rank and args.lora_alpha, "No config for LoRA"
    if args.augment_model is None: args.augment_model = args.model_name
    
    # 6. 程序入口
    print(args)
    main(args)