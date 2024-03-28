
import torch
from prompt_utils import get_prompt
import json
import argparse
import utils
from prompt_utils import *
import pdb
from data_loader import BatchDatasetLoader
from tqdm import tqdm
import openai
import time
time.sleep(10)
openai_key = 'your_openai_token'

parser = argparse.ArgumentParser()

parser.add_argument("--model", default='', type=str)
parser.add_argument("--dataset", required=True, choices=['gsm8k', 'svamp', 'math', 'numglue', 'deepmind', 'simuleq'], type=str)

parser.add_argument("--shots", default=0, type=int)

args = parser.parse_args()
def request(
        model,
        messages,
        max_tokens=512,
        temperature=0.85,
        top_p=0.7,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        api_key=None
):
    #print(model,messages,api_key)
    openai.api_base = 'https://one-api.bltcy.top/v1'
    openai.api_key = api_key
    if type(messages) is str:
        messages = [
            {
                "role": "user",
                "content": messages
            }
        ]
    if model == "gpt-35-turbo":
        model = "gpt-3.5-turbo"
    # retry request (handles connection errors, timeouts, and overloaded API)
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                timeout=2
            )
            break
        except Exception as e:
            print(str(e))
            print("Retrying...")
            if "Limit: 3 / min" in str(e):
                time.sleep(10)
            time.sleep(2)

    generations = [gen['message']['content'].lstrip() for gen in response['choices']]
    generations = [_ for _ in generations if len(_) != 0]
    return generations

def get_gpt(content):
    messages = [
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user",
                "content": content
            }
        ]
    re = request(model=args.model, messages=messages, api_key=openai_key)
    return re[-1]
get_gpt("hello")

correct, wrong = 0, 0
cot_correct = 0
pot_correct = 0
all_correct = 0

for questions, groundtruths in tqdm(BatchDatasetLoader(args.dataset, 1)):
    # First pass to use PoT
    # processed_questions = utils.process_question_with_flan_tag(questions, args.stem_flan_type)
    # if args.stem_flan_type == 'pot_prompt' and args.cot_backup:
    #     returned_values, rerun_questions, rerun_groundtruths = run_question_answer(processed_questions, groundtruths, collect_rerun=True)
    #     if rerun_questions:
    #         processed_questions = utils.process_question_with_flan_tag(rerun_questions, "")
    #         tmp = run_question_answer(processed_questions, rerun_groundtruths, collect_rerun=False)
    #         returned_values += tmp
    # else:
    #     returned_values = run_question_answer(processed_questions, groundtruths, collect_rerun=False)
    processed_questions = utils.process_question_with_flan_tag(questions, 'pot_prompt')
    groundtruth = groundtruths[0]
    used_examples = get_examples(args.dataset, 4, 'pot_prompt')
    prompt_no_input, prefix = get_prompt(used_examples, form='alpaca')
    input_str = prompt_no_input+ prefix.format(query=processed_questions[0])
    #print(input_str)
    pot_output = get_gpt(input_str)
    #print(pot_output)
    pot_output = pot_output.replace("```python","")
    pot_output = pot_output.replace("```","")
    pot_answer = utils.execute_with_timeout(pot_output)
    pot_answer = 'The answer is' + ' ' + pot_answer
    pot_answer = utils.answer_clean(args.dataset, ('####', 'The answer is'), pot_answer)
    #print(pot_answer)

    processed_questions = utils.process_question_with_flan_tag(questions, '')
    used_examples = get_examples(args.dataset, 4, '')
    prompt_no_input, prefix = get_prompt(used_examples, form='alpaca')
    input_str = prompt_no_input+ prefix.format(query=processed_questions[0])
    #print(input_str)
    cot_output = get_gpt(input_str)
    #print(cot_output)
    cot_answer = utils.answer_clean(args.dataset, ('####', 'The answer is'), cot_output)
    #print(cot_answer)
    if pot_answer == groundtruth:
        correct += 1
    else:
        wrong += 1
    if cot_answer == groundtruth and pot_answer != groundtruth:
        cot_correct += 1
        print("only cot correct:")
        print(f"{processed_questions[0]}\n{cot_output}\n{pot_output}")
    if cot_answer != groundtruth and pot_answer == groundtruth:
        pot_correct += 1
        print("only pot correct:")
        print(f"{processed_questions[0]}\n{cot_output}\n{pot_output}")
    if cot_answer == groundtruth and pot_answer == groundtruth:
        all_correct += 1
    print(cot_answer,'#',pot_answer, '#', groundtruth, '#', correct / (correct + wrong),'#',
        cot_correct / (correct + wrong),'#',pot_correct / (correct + wrong),'#',all_correct/(correct + wrong))


print('{} final accuracy {} \n {} {} {}\n-- model is {}'.format(args.dataset,correct / (correct + wrong),cot_correct / (correct + wrong),
                                                                    pot_correct / (correct + wrong),all_correct/(correct + wrong),args.model))



