import re


def resolve_eos_ids(tokenizer, model=None):
    eos_set = set()

    # from model.generation_config
    if model is not None and getattr(model, "generation_config", None) is not None:
        val = model.generation_config.eos_token_id
        if val is not None:
            if isinstance(val, list):
                eos_set.update(val)
            else:
                eos_set.add(val)

    # from model.config
    if model is not None and getattr(model, "config", None) is not None:
        val = model.config.eos_token_id
        if val is not None:
            if isinstance(val, list):
                eos_set.update(val)
            else:
                eos_set.add(val)

    # from tokenizer
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos_set.add(tokenizer.eos_token_id)

    # from common special tokens
    candidates = {"</s>", "<eos>", "<|eos|>", "<im_end>", "<|im_end|>", "<|eot_id|>", "<|endoftext|>"}
    if getattr(tokenizer, "chat_template", None):
        candidates.update(re.findall(r"<\|?[\w]+?\|?>", tokenizer.chat_template))

    for tok in candidates:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid >= 0 and tid != tokenizer.unk_token_id:
                eos_set.add(tid)
        except Exception:
            pass

    return sorted(eos_set)