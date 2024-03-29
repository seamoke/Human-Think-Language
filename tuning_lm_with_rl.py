import os

import torch
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
import pdb
from transformers import (
    Adafactor,
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser,
    pipeline
)
import extract_utils 
from trl import PPOConfig, PPOTrainer, set_seed,AutoModelForCausalLMWithValueHead
#from modeling_value_head import 
from trl.core import LengthSampler


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    max_length: Optional[int] = field(default=512, metadata={"help": "maximum length for input"})
    output_max_length: Optional[int] = field(default=512, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="./checkpoints/tuning_llama_rl/",
                                      metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

set_seed(script_args.seed)


PROMPT_COT_TO_POT = {
    "prompt_insert": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "After the instruction, there is an existing solution. You need to write a corresponding solution program when referring to this solution."
        "### Instruction:\n{instruction}\n"
        "### Solution:\n{cot}\n\n"
        "Let's write a program.\n### Response:"
    ),
    "prompt_all": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "I'd like you to solve this problem in 3 steps:"
        "1.Answer the question in plain language without writing any code.\n"
        "2.Output one line of \"*\"\n."
        "3.Write program code based on the solution process in step 1 to solve the problem.\n"
        "### Instruction:\n{instruction}\nLet's write a program.\n### Response:"
    ),
}
prompt_insert, prompt_all = PROMPT_COT_TO_POT["prompt_insert"], PROMPT_COT_TO_POT["prompt_all"]
# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.

def get_cot_answer(target):
    answer = extract_utils.answer_clean("gsm8k", ('####', 'The answer is'), target)
    return answer

def get_pot_answer(target):
    target = target.split("### Instruction")[0]
    tmp = extract_utils.execute_with_timeout(target)
    tmp = 'The answer is' + ' ' + tmp
    answer = extract_utils.answer_clean("gsm8k", ('####', 'The answer is'), tmp)
    return answer
def build_dataset(
        tokenizer, dataset_name, input_min_text_length=2, input_max_text_length=8
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    print(dataset_name)
    train_dataset = load_dataset(path="json",data_files=dataset_name,split='train')
    
    print(train_dataset)
    original_columns = train_dataset.column_names
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "groundtruth": [],
            "input_ids": []
        }
        for question,output in zip(examples['question'],examples['output']):
            query = prompt_all.format(instruction=question)
            output = output.split('****')[0]
            output = get_cot_answer(output)
            #query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples['groundtruth'].append(output)
            new_examples["input_ids"].append(tokenized_question["input_ids"])
            

        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

    ds = ds.filter(lambda x: len(x["input_ids"]) < script_args.max_length, batched=False)

    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

reward_model_name = script_args.reward_model_name
config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    init_kl_coef=0.01,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
rw_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 16,
    "truncation": True
}

if "decapoda" in script_args.model_name.lower():
    tokenizer = LlamaTokenizer.from_pretrained(script_args.model_name)
    # required for llama
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer, script_args.dataset_name)
# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index

lora_config = LoraConfig(  
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=True,
    peft_config=lora_config,
    device_map={"": current_device},
    use_cache=False
)

optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug
# reward_model = pipeline(
#     "text-classification",
#     model=reward_model_name,
#     device_map={"": current_device},
#     model_kwargs={"load_in_8bit": True},
#     tokenizer=tokenizer,
# )

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,
}
output_min_length = 32
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)

def reward_function(target,groundtruth):
    cot_output,pot_output = target.split('****')[0],target.split('****')[-1]
    cot_anwser = get_cot_answer(cot_output)
    pot_anwser = get_pot_answer(pot_output)
    cot_cha,pot_cha = 100000000,100000000
    exp = 0.001
    try:
        cot_cha = eval(f"abs({cot_anwser}-{groundtruth})")
        pot_cha = eval(f"abs({pot_anwser}-{groundtruth})")
    except Exception:
        pass

    if cot_anwser == "" and pot_anwser == "":
        return -1
    if cot_anwser == "" or pot_anwser == "":
        return -0.5
    if (cot_anwser == groundtruth or cot_cha < exp ) and pot_anwser != groundtruth:
        return 0.1
    if cot_anwser != groundtruth and (pot_anwser == groundtruth or pot_cha < exp ):
        return 0.5
    if (cot_anwser == groundtruth or cot_cha < exp ) and (pot_anwser == groundtruth or pot_cha < exp ):
        return 1
    return 0.0

train_epochs = 3
for x in range(train_epochs):
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader),desc='Train',total=len(ppo_trainer.dataloader)):

        question_tensors = batch["input_ids"]
        groundtruths = batch['groundtruth']
        
        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        # Compute sentiment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        # reward_outputs[] = reward_model(texts, **rw_kwargs)
        rewards = []
        for output,groundtruth in zip(texts,groundtruths):
            rewards.append(torch.tensor(reward_function(output,groundtruth),dtype=torch.float16))
        #rewards = [torch.tensor(output[0]["score"] - script_args.reward_baseline) for output in reward_outputs]
        # Run PPO step
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
            ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
ppo_trainer.save_pretrained(script_args.output_dir + "final")