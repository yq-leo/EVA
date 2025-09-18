import json
import sys
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, AutoModelForSeq2SeqLM, AutoConfig
import torch
sys.path.append('/home/qiyu6/EVA')
# import model_info
# from model_info import inst_cot_prompts
# import task_info
from task_info import get_test_df, clean_answer
from eos import resolve_eos_ids


def resolve_internlm_eos_ids(tokenizer, model=None):
    """
    Return a list of EOS/stop token IDs for InternLM-style chat models.
    Includes </s> and chat stop tokens like <im_end> / <|im_end|> if present.
    """
    eos_ids = set()

    # 1) tokenizer's own eos
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos_ids.add(int(tokenizer.eos_token_id))

    # 2) common stop tokens used by chat templates
    candidates = [
        "</s>", "<eos>", "<|eos|>",
        "<im_end>", "<|im_end|>",  # InternLM/Qwen-style
        "<|eot_id|>", "<|endoftext|>", "<|end_of_text|>", "eos_token"
    ]

    # 3) anything listed as a special/added token on the tokenizer
    specials = []
    for attr in ["special_tokens_map", "special_tokens_map_extended", "added_tokens_decoder"]:
        obj = getattr(tokenizer, attr, None)
        if obj:
            specials.extend(list(obj.keys()) if isinstance(obj, dict) else obj)

    # 4) parse chat_template for literal tokens like <im_end>
    chat_tpl = getattr(tokenizer, "chat_template", None)
    if isinstance(chat_tpl, str):
        specials.extend(re.findall(r"<\|?[\w]+?\|?>", chat_tpl))  # e.g., <im_end>, <|im_end|>

    # merge and deduplicate
    for tok in set(candidates + specials):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != tokenizer.unk_token_id and tid >= 0:
                eos_ids.add(int(tid))
        except Exception:
            pass

    # Fallback: some tokenizers store eos as a string only
    if not eos_ids and getattr(tokenizer, "eos_token", None):
        tid = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        if tid is not None and tid != tokenizer.unk_token_id:
            eos_ids.add(int(tid))

    eos_ids = sorted(eos_ids)

    # Optional: sync to model.generation_config
    if model is not None and eos_ids:
        # transformers accepts int or list[int] for eos_token_id; list lets any-of stop
        cfg = getattr(model, "generation_config", None)
        if cfg is not None:
            cfg.eos_token_id = eos_ids if len(eos_ids) > 1 else eos_ids[0]
        elif getattr(model, "config", None) is not None:
            model.config.eos_token_id = eos_ids[0]

    return eos_ids


def show_eos_tokens(tokenizer, eos_ids):
    """
    Pretty-print EOS/stop token ids and their string forms.
    """
    for tid in eos_ids:
        try:
            tok = tokenizer.convert_ids_to_tokens(tid)
        except Exception:
            tok = "<UNK>"
        print(f"id={tid:<6} token={tok}")


if __name__ == "__main__":
    gsm_res_file = "/home/qiyu6/EVA/ensemble/results/GSM8K/test/InternLM7b-OpenChat-drop3-top320-filter/vanilla/pred.jsonl"
    with open(gsm_res_file, "r") as f:
        lines = f.readlines()
        first_line = json.loads(lines[3])
        print(first_line.keys())

        # print(first_line['main_model_input'])
        print(first_line['full_text'])

    # model_path = "internlm/internlm2_5-7b-chat"
    # config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16,trust_remote_code=True, low_cpu_mem_usage=True).to('cuda:6')
    # tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', truncation_side='left', trust_remote_code=True)
    # eos_ids = resolve_eos_ids(tokenizer, model)
    # print("Resolved EOS/stop ids:", eos_ids)
    # show_eos_tokens(tokenizer, eos_ids)
