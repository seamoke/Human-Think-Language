# Load model directly
import torch
from prompt_utils import get_prompt
import json
import argparse
import utils
from prompt_utils import *
import pdb
from data_loader import BatchDatasetLoader
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='', type=str)
parser.add_argument("--output", default='', type=str)
parser.add_argument("--stem_flan_type", default='', choices=['', 'pot_prompt'], type=str)
parser.add_argument("--dtype", default='bfloat16', type=str)
parser.add_argument("--dataset", required=True, choices=['gsm8k', 'svamp', 'math', 'numglue', 'deepmind', 'simuleq'], type=str)
parser.add_argument("--use_vllm", action='store_true', default=False)
parser.add_argument("--load_8bit", action='store_true', default=False)
parser.add_argument("--form", default='alpaca', type=str)
parser.add_argument("--shots", default=0, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--gpus", default=4, type=int)
parser.add_argument("--print", action='store_true', default=False)
parser.add_argument("--model_max_length", default=1024, type=int)
parser.add_argument("--cot_backup", action='store_true', default=False)

args = parser.parse_args()

DTYPES = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}


if __name__ == "__main__":

    correct, wrong = 0, 0
    cot_correct = 0
    pot_correct = 0
    all_correct = 0

    for questions, groundtruths in tqdm(BatchDatasetLoader(args.dataset, args.batch_size)):
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

        processed_questions = utils.process_question_with_flan_tag(questions, args.stem_flan_type)
        used_examples = get_examples(args.dataset, args.shots, 'pot_prompt')
  
        #pot_returned_values = run_question_answer(processed_questions, groundtruths, collect_rerun=False, stem_flan_type= 'pot_prompt')
        print("pot finished")
        processed_questions = utils.process_question_with_flan_tag(questions, "")
        used_examples = get_examples(args.dataset, args.shots, '')
        #cot_returned_values = run_question_answer(processed_questions, groundtruths, collect_rerun=False, stem_flan_type="")
        print("cot finished")
        returned_values = []
        for i in range(len(pot_returned_values)):
            # 
            groundtruth = pot_returned_values[i][3]
            question,pot_output,pot_answer = pot_returned_values[i][0],pot_returned_values[i][1], pot_returned_values[i][2]
            cot_output,cot_answer = cot_returned_values[i][1],cot_returned_values[i][2]
            returned_values.append((cot_answer,pot_answer,groundtruth,question,cot_output,pot_output))

 
        for cot_answer,pot_answer,groundtruth,question,cot_output,pot_output in returned_values:
            if args.dataset == 'math':
                assert len(groundtruth) == 2, groundtruth
                groundtruth_str, groundtruth_num = groundtruth
                if utils.compare_both_string_and_number_format(pot_answer, groundtruth_str, groundtruth_num):
                    correct += 1
                else:
                    wrong += 1
            else:
                if pot_answer == groundtruth:
                    correct += 1
                else:
                    wrong += 1
            if cot_answer == groundtruth and pot_answer != groundtruth:
                cot_correct += 1
                print("only cot correct:")
                print(f"{question}\n{cot_output}\n{pot_output}")
            if cot_answer != groundtruth and pot_answer == groundtruth:
                pot_correct += 1
                print("only pot correct:")
                print(f"{question}\n{cot_output}\n{pot_output}")
            if cot_answer == groundtruth and pot_answer == groundtruth:
                all_correct += 1
            if args.print:
                print(cot_answer,'#',pot_answer, '#', groundtruth, '#', correct / (correct + wrong),'#',
                      cot_correct / (correct + wrong),'#',pot_correct / (correct + wrong),'#',all_correct/(correct + wrong))

            example = {
                'correct': groundtruth
            }

            file_handle.write(json.dumps(example) + '\n')

    print('{} final accuracy {} \n {} {} {}\n-- model is {}'.format(args.dataset,correct / (correct + wrong),cot_correct / (correct + wrong),
                                                                    pot_correct / (correct + wrong),all_correct/(correct + wrong),args.model))
    file_handle.close()
