import sys
# root="/home/qiyu6/EVA"
# sys.path.append(root)
# import model_info
# from model_info import models, module_names, vocab_sizes
from transformers import AutoModelForCausalLM
import argparse
import json


def get_name_of_head_weight(model):
    head = model.get_output_embeddings()
    name_of_head_weight = None
    target_id = head.weight.data_ptr()
    for n, p in model.named_parameters():
        if p.data_ptr() == target_id:
            name_of_head_weight = n
            break
    return name_of_head_weight


def main(args):
    llms_config = json.load(open("confs/LLMs.json"))

    model_name=args.model
    model_ckpt = llms_config[model_name]["model"]
    # specific_module_name = module_names[model_name]
    model = AutoModelForCausalLM.from_pretrained(model_ckpt, trust_remote_code=True,low_cpu_mem_usage=True)
    specific_module_name = get_name_of_head_weight(model)
    emb_shape = model.get_input_embeddings().weight.shape
    print("The shape of embedding matrix of", model_name, "is", emb_shape)
    specific_module_parameters = model.state_dict()[specific_module_name].numpy()

    save_file = "map_file/"
    with open(save_file+ model_name +".emb","w") as f:
        f.write(str(emb_shape[0]) + " " + str(emb_shape[1]) + "\n")
        for i in range(emb_shape[0]):
            token = str(i)
            vector = specific_module_parameters[i]
            f.write(token + " " + " ".join(map(str, vector.flatten())) + "\n")

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)

args = parser.parse_args()

main(args)









