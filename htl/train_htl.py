#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import os
import json

import datasets
import pdb
import torch
import numpy as np
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import pathlib
from new_llama import LlamaForCausalLM
import utils
import re
import random
#from trl import SFTTrainer
#os.system("echo $PYTORCH_CUDA_ALLOC_CONF")

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_Pause_TOKEN = "<pause>"
DEFAULT_text_TOKEN = "<text>"
DEFAULT_Code_TOKEN = "<Code>"
padding_length = 160

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    padding_side: Optional[str] = field(default="right")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    template_variation: bool = field(
        default=True, metadata={"help": "whether to use template variation"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    flash_attn: bool = field(default=False)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)

    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def get_data(data_path: str, tokenizer: transformers.PreTrainedTokenizer, template_variation: bool):
    logging.warning("Loading data...")

    if os.path.exists(data_path):
        list_data_dict = []
        with open(data_path, 'r') as f:
            for line in f.readlines():
                list_data_dict.append(json.loads(line))
        logging.warning("Formatting inputs...")

        PROMPT_DICT = utils.PROMPT_COT_TO_POT
        prompt_insert, prompt_all = PROMPT_DICT["prompt_insert"], PROMPT_DICT["prompt_all"]

        # aim_data = []
        # for x in list_data_dict:
        #     if _tokenize_fn([prompt_all.format(instruction=x['question'])+x['output']], tokenizer)['input_ids_lens'][0] < 800:
        #         aim_data.append(x)

        # print(len(aim_data))
        # with open("./math_eval/dataset/mistral/mistral_data_75k.jsonl", "w") as f:
        #     for d in aim_data:
        #         f.write(json.dumps(d) + "\n") 
        # pdb.set_trace()
        sources,targets = [],[]
        for x in list_data_dict:
            sources.append(prompt_all.format(instruction=x['question']))
            x['output'] = x['output'].replace('****',DEFAULT_Code_TOKEN)
            targets.append(f"{DEFAULT_text_TOKEN}{x['output']}{tokenizer.eos_token}")

        data_list = list(range(len(sources)))
        # 使用 numpy 生成随机采样的索引
        data_list = np.random.choice(data_list, int(len(data_list)* 0.3), replace=False)
        data_list = [list_data_dict[id] for id in data_list]
        insert_sources = [prompt_insert.format(instruction=x['question'], cot=x['output'].split(DEFAULT_Code_TOKEN)[0]) for x in data_list]
        insert_targets = [f"{x['output'].split(DEFAULT_Code_TOKEN)[-1]}{tokenizer.eos_token}" for x in data_list]
        sources += insert_sources
        targets += insert_targets

        # for id in data_list:
        #     x = list_data_dict[id]
        #     cot, pot = x['output'].split('****')
        #     sources.append(prompt_insert.format(instruction=x['question'], cot=cot))
        #     targets.append(f"{pot}{tokenizer.eos_token}")

    else:
        list_data_dict = datasets.load_dataset(data_path)["train"]

        logging.warning("Formatting inputs...")
        if template_variation:
            PROMPT_DICT = random.choice(utils.PROMPT_TEMPLATE)
        else:
            PROMPT_DICT = utils.PROMPT_TEMPLATE_SINGLE
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

        sources = []
        targets = []
        for example in list_data_dict:
            if "PoT" in example['source']:
                continue
                # text = example['instruction']
                # matches = re.findall("please|Let's", text, re.IGNORECASE)
                # lastest_postion = text.rfind(matches[-1])
                # example['instruction'] = text[0:lastest_postion]
            if 'The answer is' in example['output']:
                if example.get("input", "") != "":
                    sources.append(prompt_input.format_map(example))
                else:
                    sources.append(prompt_no_input.format_map(example)) 
                answer = example['output'].split('The answer is')[-1]
                answer = f'The answer is{answer}{tokenizer.eos_token}'
                targets.append(answer)
        #targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
        targets = [f"{DEFAULT_Pause_TOKEN*50}{x}" for x in targets]
        #targets = [f"{DEFAULT_Pause_TOKEN*50}{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
    add_pause_to_answer = False
    if add_pause_to_answer:
        temp = []
        sentences = []
        for example in targets:
            example = example.replace('\r\n','\n')
            t = example.split('\n')
            sentences.append(len(t))
            temp.append(f'{DEFAULT_Pause_TOKEN*5}'+f'{DEFAULT_Pause_TOKEN*5}\n'.join(t))
        arr = np.array(sentences)
        print(f"max: {np.max(arr)}\nmin: {np.min(arr)} \nmean: {np.mean(arr)}")
        print(f"mediem: {np.median(arr)}\nquantiles: {np.percentile(arr, [25, 75])}")
        targets = temp
    data_list = list(range(len(sources)))
    random.shuffle(data_list)
    sources = [sources[x] for x in data_list]
    targets = [targets[x] for x in data_list]

    # sentences = [_tokenize_fn([sources[x]+targets[x]], tokenizer)['input_ids_lens'][0] for x in data_list[0:10000]]
    # pdb.set_trace()
    # max_length0 = 0
    # count = 0
    # for x in range(len(sources)):
    #     if(len(sources[x])+len(targets[x]))>2048:
    #         #print(sources[x],targets[x])
    #         #print('*'*50)
    #         count += 1
    #     max_length0 = max(max_length0,len(sources[x])+len(targets[x]))
    # print(f"max_length:{count}")
    #pdb.set_trace()
    # val_split = [_tokenize_fn([strings], tokenizer) for strings in sources[0:100]]

    # print(min(val_split))

    ratio = int(0.9 * len(sources))
    print(sources[0])
    print("*"*30)
    print(targets[0])
    train_data = [sources[0:ratio],targets[0:ratio]]
    eval_data = [sources[ratio:],targets[ratio:]]
    return train_data,eval_data

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data):
        super(SupervisedDataset, self).__init__()
        
        self.sources = data[0]
        self.targets = data[1]

    def __len__(self):
        return len(self.sources)

    def naive__getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __getitem__(self, i):
        return dict(input_ids=self.sources[i], labels=self.targets[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def naive__call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        for instance in instances:
            source = instance['input_ids']
            target = instance['labels']
            sources.append(source)
            targets.append(target)

        data_dict = preprocess(sources, targets, self.tokenizer)
        input_ids, labels = data_dict['input_ids'], data_dict['labels']
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_data,eval_data = get_data(tokenizer=tokenizer, data_path=data_args.data_path,
                                      template_variation=data_args.template_variation)
    train_dataset,eval_dataset = SupervisedDataset(train_data),SupervisedDataset(eval_data)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def train():
    transformers.logging.set_verbosity_info()
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print('Start Loading Model')
    if training_args.flash_attn:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        ).to('cuda')
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation='eager',
        ).to('cuda')
        # model = transformers.AutoModelForCausalLM.from_pretrained(
        #     model_args.model_name_or_path,
        #     cache_dir=training_args.cache_dir,
        #     #attn_implementation='eager'
        # ).to('cuda')
    print(model)
    print('Start building tokenizer')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
    )

    print("*"*50)
    print("Before adding, tokenizer length: ",len(tokenizer))
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    #tokenizer.add_tokens([DEFAULT_text_TOKEN,DEFAULT_Code_TOKEN])
    special_tokens_dict['additional_special_tokens'] = [DEFAULT_text_TOKEN,DEFAULT_Code_TOKEN] #32017 #32018
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    print("*"*50)
    print("After adding, tokenizer length: ",len(tokenizer))

    print('Start building data module')
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    worker_nums = int(os.getenv('WORKER_NUM'))
    num_gpus = int(os.getenv('NUM_GPUS'))
    print(f"worker_nums:{worker_nums}\nevery_gpus:{num_gpus}")
    total_batch = training_args.per_device_train_batch_size*num_gpus*worker_nums
    #total_batch = training_args.gradient_accumulation_steps*training_args.per_device_train_batch_size*8
    total_step = training_args.num_train_epochs*len(data_module['train_dataset'].sources)//total_batch
    model.set_total_step(total_step)
    print(f"total_step:{total_step}")
    print('Start building the trainer module')
    
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
