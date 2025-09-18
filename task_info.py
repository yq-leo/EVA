from datasets import load_from_disk
import json
import os
import re


#最大生成长度
# max_new_tokens = {
#     "nq": 10,
#     "triviaqa": 10,
#     # "e2e":64,
#     # "addsub":256,
#     # "asdiv":256,
#     "gsm8k":512,
#     # "flores":128,
# }


#数据集加载
# def get_test_df(task, run_mode, task_config):
#     if task == 'NQ':
#         test_df = []
#         with open(task_config['file_path'][f'{run_mode}_file_path'], "r", encoding="utf-8") as f:
#             for line in f:
#                 data = json.loads(line)
#                 q = data.get("question")
#                 a = data.get("answer")  # could be str or list
#                 test_df.append({"question": q, "answers": a})
#         # test_data = load_from_disk("datasets/nq/validation")
#         # test_df = []
#         # for line in test_data:
#         #     test_df.append({"question":line['question'],"answers":line['answer']})
#     elif task == "triviaqa":
#         test_data = load_from_disk("/data/xyyf/102-vocab/dataset/trivia_qa/rc")['validation']
#         test_df = []
#         for line in test_data:
#             test_df.append({"question":line['question'],"answers":line['answer']['aliases']})
#     elif task == "addsub":
#         test_data = load_from_disk("/data/xyyf/102-vocab/dataset/allenai/lila")['test']
#         test_data = test_data.filter(lambda example: example['dataset'] == 'addsub.json')
#         test_df = []
#         for line in test_data:
#             test_df.append(line)
#     elif task == "asdiv":
#         test_data = load_from_disk("/data/xyyf/102-vocab/dataset/allenai/lila")['test']
#         test_data = test_data.filter(lambda example: example['dataset'] == 'asdiv.json')
#         test_df = []
#         for line in test_data:
#             test_df.append(line)
#     elif task == 'gsm8k':
#         with open(os.path.join("/data/xyyf/001-corpus/gsm8k/grade-school-math-master/grade_school_math/data/test.jsonl"), "r", encoding="utf-8") as f:
#             test_df = []
#             for line in f:
#                 test_df.append(json.loads(line))
#     elif task == 'e2e':
#         test_data = load_from_disk("/data/xyyf/102-vocab/dataset/e2e_nlg")['test']
#         test_df = []
#         for line in test_data:
#             test_df.append({"concepts":line['meaning_representation'],"target":line['human_reference']})
            
#     return test_df


def get_test_df(task, run_mode, task_config):
    test_df = []

    if task == "MMLU":
        input_dir = task_config['file_path'][f'{run_mode}_file_path']
        input_file_list = os.listdir(input_dir)
        input_file_list.sort()
        for input_file in input_file_list:
            if not input_file.endswith('.jsonl'):
                continue
            with open(os.path.join(input_dir, input_file), "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    q = data.get("question")
                    domain = data.get("domain")
                    choice_A = data.get("A")
                    choice_B = data.get("B")
                    choice_C = data.get("C")
                    choice_D = data.get("D")
                    a = data.get("answer")  # could be str or list
                    test_df.append({
                        "question": f"There is a single choice question about {domain}. Answer the question by replying A, B, C or D.\nQuestion: {q}\nA. {choice_A}\nB. {choice_B}\nC. {choice_C}\nD. {choice_D}",
                        "answers": a
                    })


    with open(task_config['file_path'][f'{run_mode}_file_path'], "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            if task == "NQ" or task == "TriviaQA" or task == "GSM8K":
                q = data.get("question")
                a = data.get("answer")  # could be str or list
                test_df.append({"question": q, "answers": a})

            elif task == "PIQA":
                q = data.get("question")
                choice_A = data.get("A")
                choice_B = data.get("B")
                a = data.get("answer")  # could be str or list
                test_df.append({
                    "question": f"{q}\nA. {choice_A}\nB. {choice_B}",
                    "answers": a
                })

            elif task == "ARC-c":
                q = data.get("question")
                choice_A = data.get("A")
                choice_B = data.get("B")
                choice_C = data.get("C")
                choice_D = data.get("D")
                a = data.get("answer")  # could be str or list
                test_df.append({
                    "question": f"Answer the question by replying A, B, C or D.\nQuestion: {q}\nA. {choice_A}\nB. {choice_B}\nC. {choice_C}\nD. {choice_D}",
                    "answers": a
                })

            else:
                raise ValueError(f"Unsupported task in get_test_df: {task}")
            
    return test_df


def clean_answer(task, input_text):
    if task in ["NQ", "TriviaQA", "PIQA", "ARC-c", "MMLU"]:
        clean_text = input_text.strip().split('\n')[0].split('<eoa>')[0].strip()

    elif task == 'GSM8K':
        try:
            clean_text = input_text.strip().split('The answer is')[1]
            for stop_before in ["\n", "</s>", "<unk>"]:
                clean_text = clean_text.split(stop_before)[0].strip()
        except:
            clean_text = ""
            
    elif task == 'e2e':
        clean_text = input_text.strip().split('\n')[0].split('<eoa>')[0].strip()
    elif task == 'addsub' or task == 'asdiv' or task == 'gsm8k':
        INVALID_ANS = "[invalid]"
        ANSWER_TRIGGER = "The answer is"
        input_text = input_text.lower()
        preds = input_text.split(ANSWER_TRIGGER.lower())
        answer_flag = True if len(preds) > 1 else False
        if answer_flag:
            # Pick first answer with flag
            pred = preds[1]
        else:
            # Pick last number without flag
            pred = preds[-1]
        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
        if len(pred) == 0:
            return INVALID_ANS
        if answer_flag:
            # choose the first element in list
            pred = pred[0]
        else:
            # choose the last element in list
            pred = pred[-1]
        # (For arithmetic tasks) if a word ends with period, it will be omitted ...
        if pred[-1] == ".":
            pred = pred[:-1]
        clean_text = pred
    return clean_text
