import argparse
import os
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM, AutoConfig
import scipy
import scipy.sparse as sp
import torch.nn.functional as F
import json
import sys
sys.path.append('/home/qiyu6/EVA')
# import model_info
# from model_info import inst_cot_prompts
# import task_info
from task_info import max_new_tokens,get_test_df, clean_answer
from tqdm import tqdm

device0 = 'cuda:0'
device1 = 'cuda:1'

prompt_key_dict = {
    "nq":'question',
    "triviaqa":'question',
    "addsub":'input',
    "asdiv":'input',
    "gsm8k":'question',
    "e2e":'concepts',
}

short2lang = {
    'eng': 'English',
    'zho_simpl': 'Chinese',
}

count_num = [0,0,0,0,0,0,0] #统计集成模型的个数

def format_example(src, tgt, s, t=None):
    prompt =  "{}:{}={}:".format(short2lang[src],s,short2lang[tgt])
    if t is not None:
        prompt += "{}\n".format(t)
    return prompt

def gen_prompt(train_df, src, tgt, k=4):
    prompt = "Translate the following sentence from "+short2lang[src]+" to "+short2lang[tgt]+".\n"
    if k == -1:
        k = len(train_df["src"])
    for i in range(k):
        prompt += format_example(src, tgt, train_df["src"][i], train_df["tgt"][i])
    return prompt


# def build_inst_prompt(task, model, question):
#     if task == "gsm8k" or task == "addsub" or task == "asdiv":
#         prompt = inst_cot_prompts[model].format_map({"instruction": question})
#         return prompt
#     prompt_schema = prompt_schemas[model]
#     model_instruction_prefix = prompt_schema["instruction_prefix"]
#     model_instruction_suffix = prompt_schema["instruction_suffix"]
#     model_input_prefix = prompt_schema["input_prefix"]
#     model_input_suffix = prompt_schema["input_suffix"]

#     if task == 'nq' or task == 'triviaqa':
#         instruction = "Please answer the following question, your answer should be as simple as possible.\n"
#         inputs = "Question: " + question
#         prompt = model_instruction_prefix + instruction + model_instruction_suffix + \
#         model_input_prefix + inputs + model_input_suffix + "Answer:"
#     elif task == 'e2e':
#         instruction = "Please describe all aspects of the restaurant in one sentence based on the following information.\n"
#         inputs = "Information: " + question
#         prompt = model_instruction_prefix + instruction + model_instruction_suffix + \
#         model_input_prefix + inputs + model_input_suffix + "Restaurant description:"
#     return prompt


def build_inst_prompt(task, model, question, llms_config):
    task = task.upper()
    conf_files = os.listdir(f"confs/{task}")
    conf_files = [file for file in conf_files if file.endswith("json")]
    assert len(conf_files) > 0, f"No config file found for task {task}"
    conf_file = conf_files[0]

    with open(f"confs/{task}/{conf_file}", "r", encoding="utf-8") as f:
        conf = json.load(f)
    prompt_path = conf['file_path']['prompt_path']
    with open(prompt_path, "r", encoding="utf-8") as f:
        instruction = f.read()
    
    task = task.lower()
    if task == "gsm8k" or task == "addsub" or task == "asdiv":
        # prompt = inst_cot_prompts[model].format_map({"instruction": question})
        prompt = llms_config[model]["inst_cot_prompt"].format_map({"instruction": question})
        return prompt
    
    # prompt_schema = prompt_schemas[model]
    prompt_schema = llms_config[model]["prompt_schema"]
    model_instruction_prefix = prompt_schema["instruction_prefix"]
    model_instruction_suffix = prompt_schema["instruction_suffix"]
    model_input_prefix = prompt_schema["input_prefix"]
    model_input_suffix = prompt_schema["input_suffix"]

    if task == 'nq':
        inputs = "Question:" + question
        # prompt = model_instruction_prefix + instruction + model_instruction_suffix + \
        # model_input_prefix + inputs + model_input_suffix + "Answer:"
        prompt = instruction + inputs + "\nAnswer:"
    elif task == 'e2e':
        instruction = "Please describe all aspects of the restaurant in one sentence based on the following information.\n"
        inputs = "Information: " + question
        prompt = model_instruction_prefix + instruction + model_instruction_suffix + \
        model_input_prefix + inputs + model_input_suffix + "Restaurant description:"
    else:
        raise ValueError(f"Unsupported task: {task}")
    return prompt


def topk_filter(logits, top_k=40):
    filter_value = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits
    
def drop(logit, res_logits, k=5):
    flag = False
    top1_token = torch.argmax(logit, dim=-1).item()
    for res_logit in res_logits:
        topk_tokens = torch.argsort(res_logit, descending=True)[:,:k]
        if top1_token in topk_tokens:
            flag = True
    return flag


@torch.no_grad()
def eval_local(args, model, model_aux_list0, model_aux_list1,
               tokenizer, tokenizer_aux_list, sparse_matrix_list,
               test_df, dev_df=None, src=None, tgt=None, llms_config=None):

    eos_token_id = model.generation_config.eos_token_id
    predictions = []

    if args.task == 'flores':
        example_prompt = gen_prompt(dev_df, src, tgt, 4)
    
    # -------------------
    # define stop sequences
    # -------------------
    stop_strs = ["\nQuestion:", "\n\n"]   # add "\n" only if all answers are 1-line
    stop_ids_list = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strs]

    def ends_with(seq_ids, suffix_ids):
        L = len(suffix_ids)
        if L == 0: return False
        return L <= len(seq_ids) and seq_ids[-L:].tolist() == suffix_ids
    # -------------------

    for obj in tqdm(test_df):
        # -------- build prompts (strings) --------
        if args.task == 'flores':
            prompt = example_prompt + format_example(src, tgt, obj)
        else:
            prompt = build_inst_prompt(args.task, args.model, obj[prompt_key_dict[args.task]], llms_config)
            prompt_aux_list = []
            for aux_model in args.aux_models:
                prompt_aux = build_inst_prompt(args.task, aux_model, obj[prompt_key_dict[args.task]], llms_config)
                prompt_aux_list.append(prompt_aux)

        # -------- tokenize once and track token length --------
        # >>> Tokenize the main prompt ONCE and track token count
        enc = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
        input_ids = enc.input_ids.to(device0)              # shape: [1, T]
        attention_mask = enc.attention_mask.to(device0)
        prompt_len_tokens = input_ids.shape[1]             # >>> token length anchor

        # (Optional) show for debugging
        # print("prompt_len_tokens:", prompt_len_tokens)

        num_of_new_tokens = 0

        while num_of_new_tokens <= args.MAX_NEW_TOKEN:
            logits_aux_list = []
            with torch.no_grad():
                # >>> Use the token-ids tensor we've been extending
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :].to(torch.float32)
                logits = topk_filter(logits, top_k=args.topk)
                logits = F.softmax(logits, dim=-1).to('cpu')

                # ---- aux models (still string-based) ----
                len0 = len(model_aux_list0)
                for aux_idx, aux_item in enumerate(zip(tokenizer_aux_list[:len0], model_aux_list0, sparse_matrix_list[:len0])):
                    tokenizer_aux, model_aux, sparse_matrix = aux_item
                    if args.task == 'flores':
                        input_ids_aux = tokenizer_aux(prompt, return_tensors="pt").input_ids.to(device0)
                    else:
                        input_ids_aux = tokenizer_aux(prompt_aux_list[aux_idx], return_tensors="pt").input_ids.to(device0)
                    logits_aux = model_aux(input_ids=input_ids_aux).logits[:, -1, :].to(torch.float32)
                    logits_aux = topk_filter(logits_aux, top_k=args.topk)
                    logits_aux = F.softmax(logits_aux, dim=-1)
                    logits_aux = logits_aux.t()
                    logits_aux = torch.spmm(sparse_matrix.to(device0), logits_aux)
                    logits_aux = logits_aux.t().to('cpu')
                    logits_aux_list.append(logits_aux)

                for aux_idx, aux_item in enumerate(zip(tokenizer_aux_list[len0:], model_aux_list1, sparse_matrix_list[len0:])):
                    tokenizer_aux, model_aux, sparse_matrix = aux_item
                    if args.task == 'flores':
                        input_ids_aux = tokenizer_aux(prompt, return_tensors="pt").input_ids.to(device1)
                    else:
                        input_ids_aux = tokenizer_aux(prompt_aux_list[aux_idx], return_tensors="pt").input_ids.to(device1)
                    logits_aux = model_aux(input_ids=input_ids_aux).logits[:, -1, :].to(torch.float32)
                    logits_aux = topk_filter(logits_aux, top_k=args.topk)
                    logits_aux = F.softmax(logits_aux, dim=-1)
                    logits_aux = logits_aux.t()
                    logits_aux = torch.spmm(sparse_matrix.to(device1), logits_aux)
                    logits_aux = logits_aux.t().to('cpu')
                    logits_aux_list.append(logits_aux)

            # ---- ensemble ----
            if args.aux_method == "linear_sum":
                print("main model lambda:", 1 - sum(args.aux_lambda))
                ensemble_logits = (1 - sum(args.aux_lambda)) * logits
                for x, y in zip(args.aux_lambda, logits_aux_list):
                    ensemble_logits += x * y
            elif args.aux_method == "drop":
                tmp_list = logits_aux_list + [logits]
                res_list = []
                for idx, val in enumerate(tmp_list):
                    res_logits = tmp_list[:idx] + tmp_list[idx+1:]
                    if drop(val, res_logits, args.drop):
                        res_list.append(val)
                count_num[len(res_list)] += 1
                if len(res_list) == 0:
                    res_list = tmp_list
                tmp_weight = 1 / len(res_list)
                for idx, val in enumerate(res_list):
                    ensemble_logits = tmp_weight * val if idx == 0 else ensemble_logits + tmp_weight * val
            else:
                raise ValueError("error: wrong method name")

            # ---- choose next token id ----
            next_id = torch.argmax(ensemble_logits, dim=-1)        # shape [1]
            next_id_val = next_id.item()

            # EOS?
            if next_id_val == eos_token_id:
                break

            # >>> Append the TOKEN ID (not string) to input_ids
            num_of_new_tokens += 1
            next_id_tensor = torch.tensor([[next_id_val]], device=input_ids.device)
            input_ids = torch.cat([input_ids, next_id_tensor], dim=1)

            # Maintain attention mask accordingly
            attention_mask = torch.cat([attention_mask, torch.ones((1,1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)

            # check stop sequences on the fly
            gen_ids = input_ids[0, prompt_len_tokens:]
            stop_hit = any(ends_with(gen_ids, stop_ids) for stop_ids in stop_ids_list)
            if stop_hit:
                break

            # If you must keep string prompts for aux models, decode ONLY the last token cleanly
            if args.task != 'flores':
                next_token_str = tokenizer.decode([next_id_val], clean_up_tokenization_spaces=False, skip_special_tokens=False)
                for aux_idx in range(len(prompt_aux_list)):
                    prompt_aux_list[aux_idx] = prompt_aux_list[aux_idx] + next_token_str
                prompt = prompt + next_token_str  # keep for display/debug only

        # -------- final decode: slice by TOKENS, not characters --------
        # full text (prompt + gen), decoded from tokens
        full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # >>> Decode the prompt part using the ORIGINAL prompt token length
        prompt_text = tokenizer.decode(input_ids[0, :prompt_len_tokens], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # >>> Decode ONLY the generated continuation (tokens after prompt)
        gen_ids = input_ids[0, prompt_len_tokens:]
        pred_text_gen = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # --- NEW: strip at first stop sequence ---
        for stop_str in ["\nQuestion:", "\n\n"]:
            if stop_str in pred_text_gen:
                pred_text_gen = pred_text_gen.split(stop_str, 1)[0]
                break
        pred_text_gen = pred_text_gen.strip()

        if args.task == 'flores':
            # if your flores formatting requires line-first answer, keep that logic here if needed
            predictions.append(pred_text_gen.strip())
        else:
            predictions.append({
                "prompt": prompt_text,
                "pred_all": pred_text_gen,   # the continuation only
            })

    return predictions


def main(args):
    llms_config = json.load(open("confs/LLMs.json", "r", encoding="utf-8"))

    #aux模型
    tokenizer_aux_list = []
    model_aux_list0 = []
    model_aux_list1 = []
    sparse_matrix_list = []
    for id, aux_model in enumerate(args.aux_models):
        model_aux_ckpt = llms_config[aux_model]["model"]
        tokenizer_aux = AutoTokenizer.from_pretrained(model_aux_ckpt, use_fast=False, add_bos_token=False, model_max_length=4096,padding_side="left",trust_remote_code=True)
        tokenizer_aux_list.append(tokenizer_aux)
        config_aux = AutoConfig.from_pretrained(model_aux_ckpt, trust_remote_code=True)
        if id > 0:
            model_aux = AutoModelForCausalLM.from_pretrained(model_aux_ckpt, config=config_aux, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True).to(device1)
            model_aux_list1.append(model_aux)
        else:
            model_aux = AutoModelForCausalLM.from_pretrained(model_aux_ckpt, config=config_aux, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True).to(device0)
            model_aux_list0.append(model_aux)

        #加载相似度矩阵
        matrix_path = f"sparse_matrix_filter/{aux_model}-{args.model}_top-10.npz"
        # matrix_path = matrix_paths[args.model][aux_model][args.matrix_name] #保存cos矩阵的位置
        scipy_matrix = sp.load_npz(matrix_path)
        sparse_matrix = torch.sparse_coo_tensor(scipy_matrix.nonzero(), scipy_matrix.data, scipy_matrix.shape).t().to(device0).to(torch.float32)#[32000,65024]
        sparse_matrix_list.append(sparse_matrix)

    #主模型
    model_ckpt = llms_config[args.model]["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, use_fast=False,add_bos_token=False, model_max_length=4096, padding_side="left",trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_ckpt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_ckpt, config=config, torch_dtype=torch.bfloat16,trust_remote_code=True, low_cpu_mem_usage=True).to(device0)

    aux_model_str = "_".join(map(str, args.aux_models))
    if args.aux_method == 'drop':
        drop_str = str(args.drop)
    else:
        drop_str = ""
    if args.task == 'flores':
        mode_str = "{}-{}-{}-4shot-{}-{}-{}{}-top{}-{}".format(args.task, args.src, args.tgt, args.model, aux_model_str, args.aux_method, drop_str, str(args.topk), args.matrix_name)
    else:
        mode_str = "{}-{}-{}-{}{}-top{}-{}".format(args.task, args.model, aux_model_str, args.aux_method, drop_str, str(args.topk), args.matrix_name)
    print("Mode: " + mode_str)#nq-xx-xx-xx-xx-linear_sum-top320-filter-inst

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, mode_str)):
        os.makedirs(os.path.join(args.save_dir, mode_str))

    if args.task == 'flores':
        dev_df = dict()
        src = args.src
        tgt = args.tgt
        data_dir = "/data/xyyf/001-corpus/flores101"
        with open(os.path.join(data_dir, "dev", src + ".dev")) as f:
            dev_df["src"] = f.read().splitlines()[:4]
        with open(os.path.join(data_dir, "devtest", src + ".devtest")) as f:
            test_df = f.read().splitlines()
        with open(os.path.join(data_dir, "dev", tgt + ".dev")) as f:
            dev_df["tgt"] = f.read().splitlines()[:4]
        predictions = eval_local(args, model, model_aux_list0, model_aux_list1, tokenizer, tokenizer_aux_list, sparse_matrix_list, test_df, dev_df, src, tgt, llms_config)
        pred_file = os.path.join(args.save_dir, mode_str, src+'-'+tgt+'.pred')
        with open(pred_file,"w",encoding='utf-8') as f:
            for pred in predictions:
                f.write(pred+'\n')
    else:
        test_df = get_test_df(args.task)
        predictions = eval_local(args, model, model_aux_list0, model_aux_list1, tokenizer, tokenizer_aux_list, sparse_matrix_list, test_df, llms_config)
        pred_file = os.path.join(args.save_dir, mode_str, 'pred.jsonl')
        with open(pred_file, "w", encoding='utf-8') as f:
            for pred, obj in zip(predictions, test_df):
                obj["prompt"] = pred["prompt"]
                obj["pred_all"] = pred["pred_all"]
                obj["prediction"] = clean_answer(args.task, pred["pred_all"])

                row = {
                    "answers": obj.get("answers"),
                    "prediction": obj.get("prediction"),
                    "question": obj.get("question"),
                    "prompt": obj.get("prompt"),
                    "pred_all": obj.get("pred_all"),   
                }

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                
    if args.aux_method == 'drop':
        with open(os.path.join(args.save_dir, mode_str, 'count_num.txt'),'w',encoding='utf-8') as f:
            f.write(str(count_num[0])+'\t'+str(count_num[1])+'\t'+str(count_num[2])+'\t'+str(count_num[3])+'\t'+str(count_num[4])+'\t'+str(count_num[5])+'\t'+str(count_num[6])+'\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--task", type=str, default='nq')
    parser.add_argument("--model", "-m", type=str, default="llama2")
    parser.add_argument("--aux_models", nargs="+", type=str, help="A list of models")
    parser.add_argument("--aux_method", type=str, default="linear_sum")
    parser.add_argument("--matrix_name", type=str, default="filter")
    parser.add_argument("--topk", type=int, default=320, help="A list of lambdas")
    parser.add_argument("--drop", type=int, default=5)
    parser.add_argument("--aux_lambda", nargs="+", type=float, help="A list of lambdas")
    parser.add_argument("--save_dir", type=str, default="/home/qiyu6/EVA/ensemble/results")
    parser.add_argument("--src", type=str, default="zho_simpl")
    parser.add_argument("--tgt", type=str, default="eng")

    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, args.task)
    args.MAX_NEW_TOKEN = max_new_tokens[args.task]
    if args.aux_lambda == None:
        if args.aux_method != "drop":
            num_of_models = len(args.aux_models)+1
            args.aux_lambda = [1/num_of_models for _ in range(num_of_models)]
        else:
            args.aux_lambda = None
    main(args)
